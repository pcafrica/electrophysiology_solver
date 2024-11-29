#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

using namespace dealii;

static inline constexpr unsigned int dim = 3;

class Common
{
public:
  Common()
  {
    if (!_initialized)
      {
        mpi_comm = MPI_COMM_WORLD;
        mpi_rank = Utilities::MPI::this_mpi_process(mpi_comm);
        mpi_size = Utilities::MPI::n_mpi_processes(mpi_comm);

        pcout_ptr =
          std::make_unique<ConditionalOStream>(std::cout, mpi_rank == 0);
        pcerr_ptr =
          std::make_unique<ConditionalOStream>(std::cerr, mpi_rank == 0);

        timer_ptr = std::make_unique<TimerOutput>(*pcout_ptr,
                                                  TimerOutput::summary,
                                                  TimerOutput::wall_times);

        _initialized = true;
      }
  }

  static inline bool _initialized = false;

  static inline MPI_Comm     mpi_comm;
  static inline unsigned int mpi_rank;
  static inline unsigned int mpi_size;

  static inline std::unique_ptr<ConditionalOStream> pcout_ptr;
#define pcout (*::Common::pcout_ptr)

  static inline std::unique_ptr<ConditionalOStream> pcerr_ptr;
#define pcerr (*::Common::pcerr_ptr)

  static inline std::unique_ptr<TimerOutput> timer_ptr;
#define timer (*::Common::timer_ptr)
};
