#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>

#include <cmath>
#include <fstream>

using namespace dealii;


namespace utils
{
  double
  heaviside_sharp(const double &x, const double &x0)
  {
    return (x > x0);
  }

  double
  heaviside(const double &x, const double &x0, const double &k)
  {
    return (0.5 * (1 + std::tanh(k * (x - x0))));
  }
} // namespace utils

// Model parameters for Bueno-Orovio
struct ModelParameters
{
  SolverControl control;

  unsigned int fe_degree                = 1;
  double       dt                       = 1e-2;
  double       final_time               = 1.;
  double       applied_current_duration = 1.;
  double       chi                      = 1;
  double       Cm                       = 1.;
  double       sigma                    = 1e-4;
  double       V1                       = 0.3;
  double       V1m                      = 0.015;
  double       V2                       = 0.015;
  double       V2m                      = 0.03;
  double       V3                       = 0.9087;
  double       Vhat                     = 1.58;
  double       Vo                       = 0.006;
  double       Vso                      = 0.65;
  double       tauop                    = 6e-3;
  double       tauopp                   = 6e-3;
  double       tausop                   = 43e-3;
  double       tausopp                  = 0.2e-3;
  double       tausi                    = 2.8723e-3;
  double       taufi                    = 0.11e-3;
  double       tau1plus                 = 1.4506e-3;
  double       tau2plus                 = 0.28;
  double       tau2inf                  = 0.07;
  double       tau1p                    = 0.06;
  double       tau1pp                   = 1.15;
  double       tau2p                    = 0.07;
  double       tau2pp                   = 0.02;
  double       tau3p                    = 2.7342e-3;
  double       tau3pp                   = 0.003;
  double       w_star_inf               = 0.94;
  double       k2                       = 65.0;
  double       k3                       = 2.0994;
  double       kso                      = 2.0;

  std::string mesh_dir = "../../meshes/idealized_lv.msh";
};



template <int dim>
class AppliedCurrent : public Function<dim>
{
public:
  AppliedCurrent(const double applied_current_duration)
    : Function<dim>()
    , p1{-0.015598, -0.0173368, 0.0307704}
    , p2{0.0264292, -0.0043322, 0.0187656}
    , p3{0.00155326, 0.0252701, 0.0248006}
  {
    t_end_current = applied_current_duration;
    p.push_back(p1);
    p.push_back(p2);
    p.push_back(p3);
  }

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

private:
  double                  t_end_current;
  std::vector<Point<dim>> p;
  Point<dim>              p1;
  Point<dim>              p2;
  Point<dim>              p3;
};



template <int dim>
void
AppliedCurrent<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<double>           &values,
                                const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}

template <int dim>
double
AppliedCurrent<dim>::value(const Point<dim> &point,
                           const unsigned int /*component*/) const
{
  const double t = this->get_time();

  static constexpr double TOL = 3e-3;

  if ((p1.distance(point) < TOL || p2.distance(point) < TOL ||
       p3.distance(point) < TOL) &&
      (t >= 0 && t <= t_end_current))
    {
      return 300;
    }

  return 0;
}


template <int dim>
class IonicModel
{
private:
  void
  setup_problem();

  void
  assemble_time_independent_matrix();

  std::array<double, 3>
  alpha(const double u);

  std::array<double, 3>
  beta(const double u);

  std::array<double, 3>
  w_inf(const double u);

  double
  Iion(const double u_old, const std::vector<double> &w) const;

  void
  assemble_time_terms();
  void
  update_w_and_ion();
  void
  solve_w();
  void
  solve();
  void
  output_results();
  void
  compute_error() const;


  const MPI_Comm                                 communicator;
  parallel::fullydistributed::Triangulation<dim> tria;
  MappingQ<dim>                                  mapping;
  FE_Q<dim>                                      fe;
  DoFHandler<dim>                                dof_handler;
  ConditionalOStream                             pcout;
  TimerOutput                                    computing_timer;
  SparsityPattern                                sparsity;
  AffineConstraints<double>                      constraints;
  TrilinosWrappers::PreconditionAMG              amg_preconditioner;
  TrilinosWrappers::SparseMatrix                 mass_matrix;
  TrilinosWrappers::SparseMatrix                 laplace_matrix;
  TrilinosWrappers::SparseMatrix                 system_matrix;
  LinearAlgebra::distributed::Vector<double>     system_rhs;
  std::unique_ptr<Function<dim>>                 rhs_function;
  std::unique_ptr<Function<dim>>                 Iext;
  std::unique_ptr<Function<dim>>                 analytical_solution;

  std::unique_ptr<FEValues<dim>> fe_values;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  LinearAlgebra::distributed::Vector<double> solution_old;
  LinearAlgebra::distributed::Vector<double> solution;

  LinearAlgebra::distributed::Vector<double> w0_old;
  LinearAlgebra::distributed::Vector<double> w0;
  LinearAlgebra::distributed::Vector<double> w1_old;
  LinearAlgebra::distributed::Vector<double> w1;
  LinearAlgebra::distributed::Vector<double> w2_old;
  LinearAlgebra::distributed::Vector<double> w2;

  LinearAlgebra::distributed::Vector<double> ion_at_dofs;

  //   Time stepping parameters
  double       time;
  const double dt;
  const double end_time;
  const double end_time_current; // final time external application

  SolverCG<LinearAlgebra::distributed::Vector<double>> solver;
  const ModelParameters                               &param;

public:
  IonicModel(const ModelParameters &parameters);

  void
  run();
};



template <int dim>
IonicModel<dim>::IonicModel(const ModelParameters &parameters)
  : communicator(MPI_COMM_WORLD)
  , tria(communicator)
  , mapping(1)
  , fe(parameters.fe_degree)
  , dof_handler(tria)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(communicator) == 0)
  , computing_timer(communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , dt(parameters.dt)
  , end_time(parameters.final_time)
  , end_time_current(parameters.applied_current_duration)
  , solver(const_cast<SolverControl &>(parameters.control))
  , param(parameters)
{
  static_assert(dim == 3);
  time = 0;
}



template <int dim>
std::array<double, 3>
IonicModel<dim>::alpha(const double u)
{
  std::array<double, 3> a;

  a[0] = (1.0 - utils::heaviside_sharp(u, param.V1)) /
         (utils::heaviside_sharp(u, param.V1m) * (param.tau1pp - param.tau1p) +
          param.tau1p);
  a[1] =
    (1.0 - utils::heaviside_sharp(u, param.V2)) /
    (utils::heaviside(u, param.V2m, param.k2) * (param.tau2pp - param.tau2p) +
     param.tau2p);
  a[2] =
    1.0 / (utils::heaviside_sharp(u, param.V2) * (param.tau3pp - param.tau3p) +
           param.tau3p);

  return a;
}



template <int dim>
std::array<double, 3>
IonicModel<dim>::beta(const double u)
{
  std::array<double, 3> b;

  b[0] = -utils::heaviside_sharp(u, param.V1) / param.tau1plus;
  b[1] = -utils::heaviside_sharp(u, param.V2) / param.tau2plus;
  b[2] = 0;

  return b;
}



template <int dim>
std::array<double, 3>
IonicModel<dim>::w_inf(const double u)
{
  std::array<double, 3> wi;

  wi[0] = 1.0 - utils::heaviside_sharp(u, param.V1m);
  wi[1] = utils::heaviside_sharp(u, param.Vo) *
            (param.w_star_inf - 1.0 + u / param.tau2inf) +
          1.0 - u / param.tau2inf;
  wi[2] = utils::heaviside(u, param.V3, param.k3);

  return wi;
}



template <int dim>
void
IonicModel<dim>::setup_problem()
{
  TimerOutput::Scope t(computing_timer, "Setup DoFs");

  fe_values =
    std::make_unique<FEValues<dim>>(mapping,
                                    fe,
                                    QGauss<dim>(2 * fe.degree + 1),
                                    update_values | update_JxW_values |
                                      update_gradients |
                                      update_quadrature_points);
  dof_handler.distribute_dofs(fe);
  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.clear();
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             communicator,
                                             locally_relevant_dofs);

  mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, communicator);
  laplace_matrix.reinit(locally_owned_dofs,
                        locally_owned_dofs,
                        dsp,
                        communicator);
  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       communicator);

  solution_old.reinit(locally_owned_dofs, communicator);
  solution.reinit(locally_owned_dofs, communicator);
  system_rhs.reinit(locally_owned_dofs, locally_relevant_dofs, communicator);

  w0_old.reinit(locally_owned_dofs, communicator);
  w0.reinit(locally_owned_dofs, communicator);
  w1_old.reinit(locally_owned_dofs, communicator);
  w1.reinit(locally_owned_dofs, communicator);
  w2_old.reinit(locally_owned_dofs, communicator);
  w2.reinit(locally_owned_dofs, communicator);

  ion_at_dofs.reinit(locally_owned_dofs, locally_relevant_dofs, communicator);

  Iext = std::make_unique<AppliedCurrent<dim>>(end_time_current);
}


template <int dim>
double
IonicModel<dim>::Iion(const double u_old, const std::vector<double> &w) const
{
  double Iion_val =
    utils::heaviside_sharp(u_old, param.V1) * (u_old - param.V1) *
      (param.Vhat - u_old) * w[0] / param.taufi -
    (1.0 - utils::heaviside_sharp(u_old, param.V2)) * (u_old - 0.) /
      (utils::heaviside_sharp(u_old, param.Vo) * (param.tauopp - param.tauop) +
       param.tauop) -
    utils::heaviside_sharp(u_old, param.V2) /
      (utils::heaviside(u_old, param.Vso, param.kso) *
         (param.tausopp - param.tausop) +
       param.tausop) +
    utils::heaviside_sharp(u_old, param.V2) * w[1] * w[2] / param.tausi;

  Iion_val = -Iion_val;

  return Iion_val;
}



template <int dim>
void
IonicModel<dim>::update_w_and_ion()
{
  TimerOutput::Scope t(computing_timer, "Update w and ion at DoFs");

  // update w from t_n to t_{n+1} on the locally owned DoFs for all w's
  // On top of that, evaluate Iion at DoFs
  for (const types::global_dof_index i : locally_owned_dofs)
    {
      // First, update w's
      std::array<double, 3> a      = alpha(solution_old[i]);
      std::array<double, 3> b      = beta(solution_old[i]);
      std::array<double, 3> w_infs = w_inf(solution_old[i]);

      w0[i] = w0_old[i] + dt * ((b[0] - a[0]) * w0_old[i] + a[0] * w_infs[0]);

      w1[i] = w1_old[i] + dt * ((b[1] - a[1]) * w1_old[i] + a[1] * w_infs[1]);

      w2[i] = w2_old[i] + dt * ((b[2] - a[2]) * w2_old[i] + a[2] * w_infs[2]);

      // Evaluate ion at u_n, w_{n+1}
      ion_at_dofs[i] = Iion(solution_old[i], {w0[i], w1[i], w2[i]});
    }
  ion_at_dofs.update_ghost_values();
}



/*
 * Assemble the time independent block chi*c*M/dt + A
 */
template <int dim>
void
IonicModel<dim>::assemble_time_independent_matrix()
{
  TimerOutput::Scope t(computing_timer, "Assemble time independent terms");

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Loop over standard deal.II cells
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_matrix      = 0;
          cell_mass_matrix = 0;
          fe_values->reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int q_index : fe_values->quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) += param.sigma *
                                           fe_values->shape_grad(i, q_index) *
                                           fe_values->shape_grad(j, q_index) *
                                           fe_values->JxW(q_index);

                      cell_mass_matrix(i, j) +=
                        (1. / dt) * fe_values->shape_value(i, q_index) *
                        fe_values->shape_value(j, q_index) *
                        fe_values->JxW(q_index);
                    }
                }
            }

          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 laplace_matrix);
          constraints.distribute_local_to_global(cell_mass_matrix,
                                                 local_dof_indices,
                                                 mass_matrix);
        }
    }
  mass_matrix.compress(VectorOperation::add);
  laplace_matrix.compress(VectorOperation::add);
}



template <int dim>
void
IonicModel<dim>::assemble_time_terms()
{
  system_rhs = 0;

  TimerOutput::Scope t(computing_timer, "Assemble time dependent terms");

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Loop over standard deal.II cells
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_rhs = 0;

          fe_values->reinit(cell);

          const auto        &q_points  = fe_values->get_quadrature_points();
          const unsigned int n_qpoints = q_points.size();

          std::vector<double> applied_currents(n_qpoints);
          Iext->value_list(q_points, applied_currents);

          std::vector<double> ion_at_qpoints(n_qpoints);
          fe_values->get_function_values(ion_at_dofs, ion_at_qpoints);

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int q_index : fe_values->quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  cell_rhs(i) +=
                    (applied_currents[q_index] - ion_at_qpoints[q_index]) *
                    fe_values->shape_value(i, q_index) *
                    fe_values->JxW(q_index);
                }
            }

          constraints.distribute_local_to_global(cell_rhs,
                                                 local_dof_indices,
                                                 system_rhs);
        }
    }
  system_rhs.compress(VectorOperation::add);
}



template <int dim>
void
IonicModel<dim>::solve()
{
  TimerOutput::Scope t(computing_timer, "Solve");

  solver.solve(system_matrix, solution, system_rhs, amg_preconditioner);

  constraints.distribute(solution);

  pcout << "\tNumber of outer iterations: " << param.control.last_step()
        << std::endl;
}



template <int dim>
void
IonicModel<dim>::output_results()
{
  TimerOutput::Scope t(computing_timer, "Output results");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution,
                           "transmembrane_potential",
                           DataOut<dim>::type_dof_data);

  data_out.add_data_vector(w0, "gating_variable", DataOut<dim>::type_dof_data);

  Vector<float> subdomain(tria.n_active_cells());

  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();

  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches(mapping);

  const std::string filename =
    ("output_time_" + std::to_string(time) +
     Utilities::int_to_string(tria.locally_owned_subdomain(), 4));

  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(communicator);
           i++)
        {
          filenames.push_back("output_time_" + std::to_string(time) +
                              Utilities::int_to_string(i, 4) + ".vtu");
        }
      std::ofstream master_output("output_time_" + std::to_string(time) +
                                  ".pvtu");
      data_out.write_pvtu_record(master_output, filenames);
    }
}



template <int dim>
void
IonicModel<dim>::run()
{
  // Create mesh
  {
    Triangulation<dim> tria_dummy;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(tria_dummy);
    std::ifstream mesh_file(param.mesh_dir);
    grid_in.read_msh(mesh_file);

    const double scale_factor = 1e-3;
    GridTools::scale(scale_factor, tria_dummy);

    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(communicator);
    GridTools::partition_triangulation(n_ranks, tria_dummy);

    const TriangulationDescription::Description<dim, dim> description =
      TriangulationDescription::Utilities::
        create_description_from_triangulation(tria_dummy, communicator);

    tria.create_triangulation(description);
  }

  pcout << "   Number of active cells:       " << tria.n_global_active_cells()
        << std::endl;

  setup_problem();
  pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;

  solution_old = -84e-3;
  solution     = solution_old;

  w0_old = 1.;
  w0     = w0_old;

  w1_old = 1.;
  w1     = w1_old;

  w2_old = 0;
  w2     = w2_old;
  output_results();

  assemble_time_independent_matrix();
  pcout << "Assembled time independent term: done" << std::endl;

  unsigned int iter_count = 0;

  // M/dt + A
  system_matrix.copy_from(mass_matrix);
  system_matrix.add(+1, laplace_matrix);

  amg_preconditioner.initialize(system_matrix);
  pcout << "Setup multigrid: done " << std::endl;

  while (time <= end_time)
    {
      time += dt;
      Iext->set_time(time);

      update_w_and_ion();
      assemble_time_terms();

      mass_matrix.vmult_add(system_rhs,
                            solution_old); // Add to system_rhs (M/dt) * u_n

      solve();
      pcout << "Solved at t = " << time << std::endl;
      ++iter_count;

      if ((iter_count % 10 == 0) || time < param.applied_current_duration)
        output_results();

      // update solutions
      solution_old = solution;
      w0_old       = w0;
      w1_old       = w1;
      w2_old       = w2;
    }
  pcout << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  {
    ModelParameters parameters;
    parameters.control.set_tolerance(1e-10);
    parameters.control.set_max_steps(2000);

    parameters.mesh_dir = "../idealized_lv.msh";
    parameters.fe_degree                = 1;
    parameters.dt                       = 1e-4;
    parameters.final_time               = 1.0;
    parameters.applied_current_duration = 3e-3;

    IonicModel<3> problem(parameters);
    problem.run();
  }
  return 0;
}
