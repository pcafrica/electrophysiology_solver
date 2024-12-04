#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

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

class Monodomain : public Common
{
public:
  class Parameters : public ParameterAcceptor
  {
  public:
    Parameters()
      : ParameterAcceptor("Monodomain solver")
    {
      add_parameter("fe_degree", fe_degree, "Finite Element degree");
      add_parameter("dt", dt, "Time step");
      add_parameter("time_end", time_end, "Final time");
      add_parameter("sigma", sigma, "Conductivity");
    }

    int    fe_degree = 1;
    double dt        = 1e-3;
    double time_end  = 1.;

    double sigma = 1e-4;
  };

  Monodomain(const BuenoOrovio::Parameters &model_params,
             const Parameters              &solver_params);

  void
  run();

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

  BuenoOrovio                                    ionic_model;
  const Parameters                              &params;
  parallel::fullydistributed::Triangulation<dim> tria;
  MappingQ<dim>                                  mapping;
  FE_Q<dim>                                      fe;
  DoFHandler<dim>                                dof_handler;
  SparsityPattern                                sparsity;
  AffineConstraints<double>                      constraints;
  TrilinosWrappers::PreconditionAMG              amg_preconditioner;
  TrilinosWrappers::SparseMatrix                 mass_matrix_dt;
  TrilinosWrappers::SparseMatrix                 laplace_matrix;
  TrilinosWrappers::SparseMatrix                 system_matrix;
  LinearAlgebra::distributed::Vector<double>     system_rhs;
  std::unique_ptr<Function<dim>>                 Iapp;

  std::unique_ptr<FEValues<dim>> fe_values;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  LinearAlgebra::distributed::Vector<double> u_old;
  LinearAlgebra::distributed::Vector<double> u;

  //   Time stepping parameters
  double       time;
  const double dt;
  unsigned int time_step;
  const double time_end;
};



Monodomain::Monodomain(const BuenoOrovio::Parameters &model_params,
                       const Parameters              &solver_params)
  : ionic_model(model_params)
  , params(solver_params)
  , tria(mpi_comm)
  , mapping(params.fe_degree)
  , fe(params.fe_degree)
  , dof_handler(tria)
  , time(0)
  , dt(params.dt)
  , time_step(0)
  , time_end(params.time_end)
{}



void
Monodomain::setup()
{
  TimerOutput::Scope t(timer, "Setup monodomain");

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
                                             mpi_comm,
                                             locally_relevant_dofs);

  mass_matrix_dt.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_comm);
  laplace_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_comm);
  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_comm);

  u_old.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  u.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  system_rhs.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);

  Iapp = std::make_unique<AppliedCurrent>();

  ionic_model.setup(locally_owned_dofs, locally_relevant_dofs, dt);
}


/*
 * Assemble the time independent block M/dt + A
 */
void
Monodomain::assemble_time_independent_matrix()
{
  TimerOutput::Scope t(timer, "Assemble time independent terms");

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix_dt(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Loop over standard deal.II cells
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_matrix         = 0;
          cell_mass_matrix_dt = 0;
          fe_values->reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int q_index : fe_values->quadrature_point_indices())
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) += params.sigma *
                                           fe_values->shape_grad(i, q_index) *
                                           fe_values->shape_grad(j, q_index) *
                                           fe_values->JxW(q_index);

                      cell_mass_matrix_dt(i, j) +=
                        (1. / dt) * fe_values->shape_value(i, q_index) *
                        fe_values->shape_value(j, q_index) *
                        fe_values->JxW(q_index);
                    }
                }
            }

          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 laplace_matrix);
          constraints.distribute_local_to_global(cell_mass_matrix_dt,
                                                 local_dof_indices,
                                                 mass_matrix_dt);
        }
    }
  mass_matrix_dt.compress(VectorOperation::add);
  laplace_matrix.compress(VectorOperation::add);
}



void
Monodomain::assemble_time_terms()
{
  TimerOutput::Scope t(timer, "Assemble time dependent terms");

  system_rhs = 0;

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
          fe_values->get_function_values(ionic_model.Iion, ion_at_qpoints);

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

  mass_matrix_dt.vmult_add(system_rhs,
                           u_old); // Add to system_rhs (M/dt) * u_n
}



void
Monodomain::solve()
{
  TimerOutput::Scope t(timer, "Solve");

  SolverControl solver_control(1000, 1e-10);

  SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);
  solver.solve(system_matrix, u, system_rhs, amg_preconditioner);

  constraints.distribute(u);

  pcout << "\tNumber of outer iterations: " << solver_control.last_step()
        << std::endl;
}



void
Monodomain::output_results()
{
  TimerOutput::Scope t(timer, "Output results");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  //
  data_out.add_data_vector(u,
                           "transmembrane_potential",
                           DataOut<dim>::type_dof_data);

  //
  for (unsigned int i = 0; i < ionic_model.w.size(); ++i)
    {
      data_out.add_data_vector(ionic_model.w[i],
                               "w" + std::to_string(i),
                               DataOut<dim>::type_dof_data);
    }

  //
  Vector<float> subdomain(tria.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches(mapping);

  const bool export_mesh = (time_step == 0);

  const std::string basename    = "output";
  const std::string filename_h5 = basename + "_" + std::to_string(time) + ".h5";
  const std::string filename_xdmf =
    basename + "_" + std::to_string(time) + ".xdmf";
  const std::string filename_mesh =
    basename + "_" + std::to_string(0.0) + ".h5";

  DataOutBase::DataOutFilter data_filter(
    DataOutBase::DataOutFilterFlags(true, true));

  data_out.write_filtered_data(data_filter);
  data_out.write_hdf5_parallel(
    data_filter, export_mesh, filename_mesh, filename_h5, mpi_comm);

  std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(
    data_filter, filename_mesh, filename_h5, time, mpi_comm)});

  data_out.write_xdmf_file(xdmf_entries, filename_xdmf, mpi_comm);
}



void
Monodomain::run()
{
  // Create mesh
  {
    TimerOutput::Scope t(timer, "Create mesh");
    Triangulation<dim> tria_dummy;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(tria_dummy);
    std::ifstream mesh_file("../idealized_lv.msh");
    grid_in.read_msh(mesh_file);

    const double scale_factor = 1e-3;
    GridTools::scale(scale_factor, tria_dummy);

    GridTools::partition_triangulation(mpi_size, tria_dummy);

    const TriangulationDescription::Description<dim, dim> description =
      TriangulationDescription::Utilities::
        create_description_from_triangulation(tria_dummy, mpi_comm);

    tria.create_triangulation(description);
  }

  pcout << "\tNumber of active cells:       " << tria.n_global_active_cells()
        << std::endl;

  setup();
  pcout << "\tNumber of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;

  u_old = -84e-3;
  u     = u_old;

  output_results();

  assemble_time_independent_matrix();

  // M/dt + A
  system_matrix.copy_from(mass_matrix_dt);
  system_matrix.add(+1, laplace_matrix);

  amg_preconditioner.initialize(system_matrix);

  while (time <= time_end)
    {
      time += dt;
      Iapp->set_time(time);

      ionic_model.solve(u_old);
      assemble_time_terms();

      solve();
      pcout << "Solved at t = " << time << std::endl;
      ++time_step;

      if ((time_step % 10 == 0))
        output_results();

      u_old = u;
    }
  pcout << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Monodomain::Parameters  solver_params;
  BuenoOrovio::Parameters model_params;
  ParameterAcceptor::initialize("../ionic_params.prm");

  Monodomain problem(model_params, solver_params);

  problem.run();

  return 0;
}
