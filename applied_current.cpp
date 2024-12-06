#include "applied_current.hpp"

AppliedCurrent::AppliedCurrent(const Parameters &params)
  : Function<dim>()
  , params(params)
  , p1(params.p1)
  , p2(params.p2)
  , p3(params.p3)
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
