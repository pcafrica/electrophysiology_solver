#pragma once

#include "common.hpp"
#include "utils.hpp"

class BuenoOrovio : public Common
{
public:
  BuenoOrovio() = default;

  void
  setup(const IndexSet &locally_owned_dofs,
        const IndexSet &locally_relevant_dofs);

  std::array<double, 3>
  alpha(const double u);

  std::array<double, 3>
  beta(const double u);

  std::array<double, 3>
  w_inf(const double u);

  double
  Iion_0d(const double u_old, const std::vector<double> &w) const;

  void
  solve(const IndexSet                                   &locally_owned_dofs,
        const LinearAlgebra::distributed::Vector<double> &solution_old,
        const double                                     &dt);

  // private:
  LinearAlgebra::distributed::Vector<double> w0_old;
  LinearAlgebra::distributed::Vector<double> w0;
  LinearAlgebra::distributed::Vector<double> w1_old;
  LinearAlgebra::distributed::Vector<double> w1;
  LinearAlgebra::distributed::Vector<double> w2_old;
  LinearAlgebra::distributed::Vector<double> w2;

  LinearAlgebra::distributed::Vector<double> Iion;

  double V1         = 0.3;
  double V1m        = 0.015;
  double V2         = 0.015;
  double V2m        = 0.03;
  double V3         = 0.9087;
  double Vhat       = 1.58;
  double Vo         = 0.006;
  double Vso        = 0.65;
  double tauop      = 6e-3;
  double tauopp     = 6e-3;
  double tausop     = 43e-3;
  double tausopp    = 0.2e-3;
  double tausi      = 2.8723e-3;
  double taufi      = 0.11e-3;
  double tau1plus   = 1.4506e-3;
  double tau2plus   = 0.28;
  double tau2inf    = 0.07;
  double tau1p      = 0.06;
  double tau1pp     = 1.15;
  double tau2p      = 0.07;
  double tau2pp     = 0.02;
  double tau3p      = 2.7342e-3;
  double tau3pp     = 0.003;
  double w_star_inf = 0.94;
  double k2         = 65.0;
  double k3         = 2.0994;
  double kso        = 2.0;
};

void
BuenoOrovio::setup(const IndexSet &locally_owned_dofs,
                   const IndexSet &locally_relevant_dofs)
{
  TimerOutput::Scope t(timer, "Setup ionic model");

  w0_old.reinit(locally_owned_dofs, mpi_comm);
  w0.reinit(locally_owned_dofs, mpi_comm);
  w1_old.reinit(locally_owned_dofs, mpi_comm);
  w1.reinit(locally_owned_dofs, mpi_comm);
  w2_old.reinit(locally_owned_dofs, mpi_comm);
  w2.reinit(locally_owned_dofs, mpi_comm);

  w0_old = 1.;
  w0     = w0_old;

  w1_old = 1.;
  w1     = w1_old;

  w2_old = 0;
  w2     = w2_old;

  Iion.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  Iion = 0;
}


std::array<double, 3>
BuenoOrovio::alpha(const double u)
{
  std::array<double, 3> a;

  a[0] = (1.0 - utils::heaviside_sharp(u, V1)) /
         (utils::heaviside_sharp(u, V1m) * (tau1pp - tau1p) + tau1p);
  a[1] = (1.0 - utils::heaviside_sharp(u, V2)) /
         (utils::heaviside(u, V2m, k2) * (tau2pp - tau2p) + tau2p);
  a[2] = 1.0 / (utils::heaviside_sharp(u, V2) * (tau3pp - tau3p) + tau3p);

  return a;
}



std::array<double, 3>
BuenoOrovio::beta(const double u)
{
  std::array<double, 3> b;

  b[0] = -utils::heaviside_sharp(u, V1) / tau1plus;
  b[1] = -utils::heaviside_sharp(u, V2) / tau2plus;
  b[2] = 0;

  return b;
}



std::array<double, 3>
BuenoOrovio::w_inf(const double u)
{
  std::array<double, 3> wi;

  wi[0] = 1.0 - utils::heaviside_sharp(u, V1m);
  wi[1] = utils::heaviside_sharp(u, Vo) * (w_star_inf - 1.0 + u / tau2inf) +
          1.0 - u / tau2inf;
  wi[2] = utils::heaviside(u, V3, k3);

  return wi;
}

double
BuenoOrovio::Iion_0d(const double u_old, const std::vector<double> &w) const
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
      Iion[i] = Iion_0d(solution_old[i], {w0[i], w1[i], w2[i]});
    }
  Iion.update_ghost_values();

  w0_old = w0;
  w1_old = w1;
  w2_old = w2;
}
