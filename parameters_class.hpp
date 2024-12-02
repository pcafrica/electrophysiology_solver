#pragma once
#include <deal.II/base/parameter_acceptor.h>

class IonicModelParameters : public ParameterAcceptor
{
  public:
    IonicModelParameters() : ParameterAcceptor("Model Parameters")
    {
      add_parameter("V1", V1, "V1 Parameter");
      add_parameter("V1m", V1m, "V1m Parameter");
      add_parameter("V2", V2, "V2 Parameter");
      add_parameter("V2m", V2m, "V2m Parameter");
      add_parameter("V3", V3, "V3 Parameter");
      add_parameter("Vhat", Vhat, "Vhat Parameter");
      add_parameter("Vo", Vo, "Vo Parameter");
      add_parameter("Vso", Vso, "Vso Parameter");
      add_parameter("tauop", tauop, "Time constant tauop");
      add_parameter("tauopp", tauopp, "tauopp Parameter");
      add_parameter("tausop", tausop, "tausop Parameter");
      add_parameter("tausopp", tausopp, "tausopp Parameter");
      add_parameter("tausi", tausi, "tausi Parameter");
      add_parameter("taufi", taufi, "taufi Parameter");
      add_parameter("tau1plus", tau1plus, "tau1plus Parameter");
      add_parameter("tau2plus", tau2plus, "tau2plus Parameter");
      add_parameter("tau2inf", tau2inf, "tau2inf Parameter");
      add_parameter("tau1p", tau1p, "tau1p Parameter");
      add_parameter("tau1pp", tau1pp, "tau1pp Parameter");
      add_parameter("tau2p", tau2p, "tau2p Parameter");
      add_parameter("tau2pp", tau2pp, "tau2pp Parameter");
      add_parameter("tau3p", tau3p, "tau3p Parameter");
      add_parameter("tau3pp", tau3pp, "tau3pp Parameter");
      add_parameter("w_star_inf", w_star_inf, "w_star_inf Parameter");
      add_parameter("k2", k2, "k2 Parameter");
      add_parameter("k3", k3, "k3 Parameter");
      add_parameter("kso", kso, "kso Parameter");
    };
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


class SolverParameters : public ParameterAcceptor
{
  public:
    int fe_degree = 1;
    double dt = 1e-3;
    unsigned int time_step = 0;
    double time_end = 1.;
   SolverParameters() : ParameterAcceptor("Solver Parameters")
   {
     add_parameter("fe_degree", fe_degree, "fe_degree Parameter");
     add_parameter("dt", dt, "dt Parameter");
     add_parameter("time_step", time_step, "time_step Parameter");
     add_parameter("time_end", time_end, "time_end Parameter");
   }
};


