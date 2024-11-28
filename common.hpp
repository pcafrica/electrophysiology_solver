#pragma once

#include <deal.II/base/config.h>

static inline constexpr unsigned int dim = 3;

using namespace dealii;

class Common
{
public:
  Common()
    : mpi_comm(MPI_COMM_WORLD)
    , mpi_rank(Utilities::MPI::this_mpi_process(mpi_comm))
    , mpi_size(Utilities::MPI::n_mpi_processes(mpi_comm))
    , pcout(std::cout, mpi_rank == 0)
    , timer(mpi_comm, pcout, TimerOutput::summary, TimerOutput::wall_times)
  {}

  const MPI_Comm     mpi_comm;
  const unsigned int mpi_rank;
  const unsigned int mpi_size;
  ConditionalOStream pcout;
  TimerOutput        timer;
};
