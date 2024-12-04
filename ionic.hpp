#pragma once

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/lac/la_parallel_vector.h>

#include "common.hpp"
#include "utils.hpp"

class BuenoOrovio : public Common
{
public:
  friend class Monodomain;

  class Parameters : public ParameterAcceptor
  {
  public:
    Parameters()
      : ParameterAcceptor("Monodomain solver/Bueno-Orovio")
    {
      add_parameter("V1", V1);
      add_parameter("V1m", V1m);
      add_parameter("V2", V2);
      add_parameter("V2m", V2m);
      add_parameter("V3", V3);
      add_parameter("Vhat", Vhat);
      add_parameter("Vo", Vo);
      add_parameter("Vso", Vso);
      add_parameter("tauop", tauop);
      add_parameter("tauopp", tauopp);
      add_parameter("tausop", tausop);
      add_parameter("tausopp", tausopp);
      add_parameter("tausi", tausi);
      add_parameter("taufi", taufi);
      add_parameter("tau1plus", tau1plus);
      add_parameter("tau2plus", tau2plus);
      add_parameter("tau2inf", tau2inf);
      add_parameter("tau1p", tau1p);
      add_parameter("tau1pp", tau1pp);
      add_parameter("tau2p", tau2p);
      add_parameter("tau2pp", tau2pp);
      add_parameter("tau3p", tau3p);
      add_parameter("tau3pp", tau3pp);
      add_parameter("w_star_inf", w_star_inf);
      add_parameter("k2", k2);
      add_parameter("k3", k3);
      add_parameter("kso", kso);
    }

    double V1         = 0.4;
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

  static inline constexpr unsigned int N_VARS = 3;

  BuenoOrovio(const Parameters &params);

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

  const Parameters &params;
};
