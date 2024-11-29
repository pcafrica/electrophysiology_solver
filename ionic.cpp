#include "ionic.hpp"

void
BuenoOrovio::setup(const IndexSet &locally_owned_dofs,
                   const IndexSet &locally_relevant_dofs)
{
  TimerOutput::Scope t(timer, "Setup ionic model");

  for (unsigned int i = 0; i < N_VARS; ++i)
    {
      w_old[i].reinit(locally_owned_dofs, mpi_comm);
      w[i].reinit(locally_owned_dofs, mpi_comm);
      w_old[i] = 1.;
      w[i]     = w_old[i];
    }

  Iion.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  Iion = 0;
}


std::array<double, BuenoOrovio::N_VARS>
BuenoOrovio::alpha(const double u) const
{
  std::array<double, N_VARS> a;

  a[0] = (1.0 - utils::heaviside_sharp(u, V1)) /
         (utils::heaviside_sharp(u, V1m) * (tau1pp - tau1p) + tau1p);
  a[1] = (1.0 - utils::heaviside_sharp(u, V2)) /
         (utils::heaviside(u, V2m, k2) * (tau2pp - tau2p) + tau2p);
  a[2] = 1.0 / (utils::heaviside_sharp(u, V2) * (tau3pp - tau3p) + tau3p);

  return a;
}



std::array<double, BuenoOrovio::N_VARS>
BuenoOrovio::beta(const double u) const
{
  std::array<double, N_VARS> b;

  b[0] = -utils::heaviside_sharp(u, V1) / tau1plus;
  b[1] = -utils::heaviside_sharp(u, V2) / tau2plus;
  b[2] = 0;

  return b;
}



std::array<double, BuenoOrovio::N_VARS>
BuenoOrovio::w_inf(const double u) const
{
  std::array<double, N_VARS> wi;

  wi[0] = 1.0 - utils::heaviside_sharp(u, V1m);
  wi[1] = utils::heaviside_sharp(u, Vo) * (w_star_inf - 1.0 + u / tau2inf) +
          1.0 - u / tau2inf;
  wi[2] = utils::heaviside(u, V3, k3);

  return wi;
}

double
BuenoOrovio::Iion_0d(const double                                   u_old,
                     const std::array<double, BuenoOrovio::N_VARS> &w) const
{
  const double Iion_val =
    utils::heaviside_sharp(u_old, V1) * (u_old - V1) * (Vhat - u_old) * w[0] /
      taufi -
    (1.0 - utils::heaviside_sharp(u_old, V2)) * (u_old - 0.) /
      (utils::heaviside_sharp(u_old, Vo) * (tauopp - tauop) + tauop) -
    utils::heaviside_sharp(u_old, V2) /
      (utils::heaviside(u_old, Vso, kso) * (tausopp - tausop) + tausop) +
    utils::heaviside_sharp(u_old, V2) * w[1] * w[2] / tausi;

  return -Iion_val;
}

std::array<double, BuenoOrovio::N_VARS>
BuenoOrovio::solve_0d(const double                                   u_old,
                      const std::array<double, BuenoOrovio::N_VARS> &w,
                      const double                                  &dt) const
{
  std::array<double, N_VARS> w_new;

  std::array<double, 3> a      = alpha(u_old);
  std::array<double, 3> b      = beta(u_old);
  std::array<double, 3> w_infs = w_inf(u_old);

  for (unsigned int i = 0; i < N_VARS; ++i)
    {
      w_new[i] = w[i] + dt * ((b[i] - a[i]) * w[i] + a[i] * w_infs[i]);
    }

  return w_new;
}

void
BuenoOrovio::solve(
  const IndexSet                                   &locally_owned_dofs,
  const LinearAlgebra::distributed::Vector<double> &solution_old,
  const double                                     &dt)
{
  TimerOutput::Scope t(timer, "Update w and ion at DoFs");

  // update w from t_n to t_{n+1} on the locally owned DoFs for all w's
  // On top of that, evaluate Iion at DoFs
  Iion.zero_out_ghost_values();
  for (const types::global_dof_index idx : locally_owned_dofs)
    {
      std::array<double, N_VARS> w_new =
        solve_0d(solution_old[idx],
                 {{w_old[0][idx], w_old[1][idx], w_old[2][idx]}},
                 dt);

      for (unsigned int i = 0; i < N_VARS; ++i)
        {
          w[i][idx] = w_new[i];
        }

      Iion[idx] =
        Iion_0d(solution_old[idx], {{w[0][idx], w[1][idx], w[2][idx]}});
    }

  Iion.update_ghost_values();

  w_old = w;
}
