#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include "common.hpp"
#include "utils.hpp"

class BuenoOrovio : public Common
{
public:
  static inline constexpr unsigned int N_VARS = 3;

  BuenoOrovio() = default;

  void
  setup(const IndexSet &locally_owned_dofs,
        const IndexSet &locally_relevant_dofs);

  std::array<double, N_VARS>
  alpha(const double u) const;

  std::array<double, N_VARS>
  beta(const double u) const;

  std::array<double, N_VARS>
  w_inf(const double u) const;

  double
  Iion_0d(const double u_old, const std::array<double, N_VARS> &w) const;

  std::array<double, N_VARS>
  solve_0d(const double                      u_old,
           const std::array<double, N_VARS> &w,
           const double                     &dt) const;

  void
  solve(const IndexSet                                   &locally_owned_dofs,
        const LinearAlgebra::distributed::Vector<double> &solution_old,
        const double                                     &dt);

  // private:
  std::array<LinearAlgebra::distributed::Vector<double>, N_VARS> w_old;
  std::array<LinearAlgebra::distributed::Vector<double>, N_VARS> w;

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
