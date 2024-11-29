#include "applied_current.hpp"

AppliedCurrent::AppliedCurrent()
  : Function<dim>()
  , p1{-0.015598, -0.0173368, 0.0307704}
  , p2{0.0264292, -0.0043322, 0.0187656}
  , p3{0.00155326, 0.0252701, 0.0248006}
{
  p.push_back(p1);
  p.push_back(p2);
  p.push_back(p3);
}

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
      (t >= 0 && t <= 3e-3))
    {
      return 300;
    }

  return 0;
}
