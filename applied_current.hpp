#pragma once

#include <deal.II/base/function.h>

#include "common.hpp"

class AppliedCurrent : public Function<dim>
{
public:
  AppliedCurrent(const double applied_current_duration);

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  virtual void
  value_list(const std::vector<Point<dim>> &points,
             std::vector<double>           &values,
             const unsigned int /*component*/) const override;

private:
  double                  t_end_current;
  std::vector<Point<dim>> p;
  Point<dim>              p1;
  Point<dim>              p2;
  Point<dim>              p3;
};
