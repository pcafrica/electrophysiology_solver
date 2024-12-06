#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>

#include "common.hpp"

class AppliedCurrent : public Function<dim>
{
public:
  class Parameters : public ParameterAcceptor
  {
  public:
    Point<dim> p1 = {-0.015598, -0.0173368, 0.0307704};
    Point<dim> p2 = {0.0264292, -0.0043322, 0.0187656};
    Point<dim> p3 = {0.00155326, 0.0252701, 0.0248006};

    Parameters()
      : ParameterAcceptor("Monodomain solver")
    {
      enter_subsection("Applied current");
      {
        add_parameter("p1", p1);
        add_parameter("p2", p2);
        add_parameter("p3", p3);
      }
      leave_subsection();
    }
  };

  AppliedCurrent(const Parameters &params);

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

private:
  const Parameters       &params;
  std::vector<Point<dim>> p;
  Point<dim>              p1;
  Point<dim>              p2;
  Point<dim>              p3;
};
