#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#include "common.hpp"
#include "utils.hpp"
#include "parameters_class.hpp"

class BuenoOrovio : public Common
{
public:
  friend class Monodomain;

  static inline constexpr unsigned int N_VARS = 3;

  BuenoOrovio(const IonicModelParameters &params);

  void
  setup(const IndexSet &locally_owned_dofs,
        const IndexSet &locally_relevant_dofs,
        const double   &dt);

  std::array<double, N_VARS>
  alpha(const double u) const;

  std::array<double, N_VARS>
  beta(const double u) const;

  std::array<double, N_VARS>
  w_inf(const double u) const;

  double
  Iion_0d(const double u_old, const std::array<double, N_VARS> &w) const;

  std::array<double, N_VARS>
  solve_0d(const double u_old, const std::array<double, N_VARS> &w) const;

  void
  solve(const LinearAlgebra::distributed::Vector<double> &u_old);

private:
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  double dt;

  std::array<LinearAlgebra::distributed::Vector<double>, N_VARS> w_old;
  std::array<LinearAlgebra::distributed::Vector<double>, N_VARS> w;

  LinearAlgebra::distributed::Vector<double> Iion;
  const IonicModelParameters &params;
};
