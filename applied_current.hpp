#pragma once

#include "common.hpp"
#include <deal.II/base/function.h>

class AppliedCurrent : public Function<dim>
{
public:
  AppliedCurrent(const double applied_current_duration)
    : Function<dim>()
    , p1{-0.015598, -0.0173368, 0.0307704}
    , p2{0.0264292, -0.0043322, 0.0187656}
    , p3{0.00155326, 0.0252701, 0.0248006}
  {
    t_end_current = applied_current_duration;
    p.push_back(p1);
    p.push_back(p2);
    p.push_back(p3);
  }

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



void
AppliedCurrent::value_list(const std::vector<Point<dim>> &points,
                           std::vector<double>           &values,
                           const unsigned int /*component*/) const
{
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = this->value(points[i]);
}

double
AppliedCurrent::value(const Point<dim> &point,
                      const unsigned int /*component*/) const
{
  const double t = this->get_time();

  static constexpr double TOL = 3e-3;

  if ((p1.distance(point) < TOL || p2.distance(point) < TOL ||
       p3.distance(point) < TOL) &&
      (t >= 0 && t <= t_end_current))
    {
      return 300;
    }

  return 0;
}
