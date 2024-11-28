#include <deal.II/base/conditional_ostream.h>
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

#include "applied_current.hpp"
#include "common.hpp"
#include "ionic.hpp"

using namespace dealii;


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

  std::string mesh_dir = "../../meshes/idealized_lv.msh";
};



class Monodomain
{
private:
  void
  setup();

  void
  assemble_time_independent_matrix();

  void
  assemble_time_terms();
  void
  solve();
  void
  output_results();

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
  std::unique_ptr<Function<dim>>                 Iapp;

  std::unique_ptr<FEValues<dim>> fe_values;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  LinearAlgebra::distributed::Vector<double> solution_old;
  LinearAlgebra::distributed::Vector<double> solution;

  //   Time stepping parameters
  double       time;
  const double dt;
  const double end_time;
  const double end_time_current; // final time external application

  unsigned int iter_count;

  SolverCG<LinearAlgebra::distributed::Vector<double>> solver;

  const ModelParameters &param;

  BuenoOrovio ionic_model;

public:
  Monodomain(const ModelParameters &parameters);

  void
  run();
};



Monodomain::Monodomain(const ModelParameters &parameters)
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



void
Monodomain::setup()
{
  TimerOutput::Scope t(computing_timer, "Setup monodomain");

  fe_values =
    std::make_unique<FEValues<dim>>(mapping,
                                    fe,
                                    QGauss<dim>(fe.degree + 1),
                                    update_values | update_JxW_values |
                                      update_gradients |
                                      update_quadrature_points);

  dof_handler.distribute_dofs(fe);
  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
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

  solution_old.reinit(locally_owned_dofs, locally_relevant_dofs, communicator);
  solution.reinit(locally_owned_dofs, locally_relevant_dofs, communicator);
  system_rhs.reinit(locally_owned_dofs, locally_relevant_dofs, communicator);

  Iapp = std::make_unique<AppliedCurrent>(end_time_current);

  ionic_model.setup(locally_owned_dofs, locally_relevant_dofs);
}


/*
 * Assemble the time independent block chi*c*M/dt + A
 */
void
Monodomain::assemble_time_independent_matrix()
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



void
Monodomain::assemble_time_terms()
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
          Iapp->value_list(q_points, applied_currents);

          std::vector<double> ion_at_qpoints(n_qpoints);
          fe_values->get_function_values(ionic_model.ion_at_dofs,
                                         ion_at_qpoints);

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



void
Monodomain::solve()
{
  TimerOutput::Scope t(computing_timer, "Solve");

  solver.solve(system_matrix, solution, system_rhs, amg_preconditioner);

  constraints.distribute(solution);

  pcout << "\tNumber of outer iterations: " << param.control.last_step()
        << std::endl;
}



void
Monodomain::output_results()
{
  TimerOutput::Scope t(computing_timer, "Output results");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  //
  data_out.add_data_vector(solution,
                           "transmembrane_potential",
                           DataOut<dim>::type_dof_data);

  //
  data_out.add_data_vector(ionic_model.w0, "w0", DataOut<dim>::type_dof_data);
  data_out.add_data_vector(ionic_model.w1, "w1", DataOut<dim>::type_dof_data);
  data_out.add_data_vector(ionic_model.w2, "w2", DataOut<dim>::type_dof_data);

  //
  Vector<float> subdomain(tria.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches(mapping);

  const bool export_mesh = (iter_count == 0);

  const std::string basename = "output";
  const std::string filename_h5 =
    basename + "_" + std::to_string(time) + ".h5";
  const std::string filename_xdmf =
    basename + "_" + std::to_string(time) + ".xdmf";
  const std::string filename_mesh = basename + "_" + std::to_string(0) + ".h5";

  DataOutBase::DataOutFilter data_filter(
    DataOutBase::DataOutFilterFlags(true, true));

  data_out.write_filtered_data(data_filter);
  data_out.write_hdf5_parallel(
    data_filter, export_mesh, filename_mesh, filename_h5, communicator);

  std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(
    data_filter, filename_mesh, filename_h5, time, communicator)});

  data_out.write_xdmf_file(xdmf_entries, filename_xdmf, communicator);
}



void
Monodomain::run()
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

  setup();
  pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;

  solution_old = -84e-3;
  solution     = solution_old;

  iter_count = 0;
  output_results();

  assemble_time_independent_matrix();
  pcout << "Assembled time independent term: done" << std::endl;

  // M/dt + A
  system_matrix.copy_from(mass_matrix);
  system_matrix.add(+1, laplace_matrix);

  amg_preconditioner.initialize(system_matrix);
  pcout << "Setup multigrid: done " << std::endl;

  while (time <= end_time)
    {
      time += dt;
      Iapp->set_time(time);

      ionic_model.solve(locally_owned_dofs, solution_old, dt);
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

    parameters.mesh_dir                 = "../idealized_lv.msh";
    parameters.fe_degree                = 1;
    parameters.dt                       = 1e-4;
    parameters.final_time               = 1.0;
    parameters.applied_current_duration = 3e-3;

    Monodomain problem(parameters);
    problem.run();
  }
  return 0;
}
