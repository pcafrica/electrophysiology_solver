#include "utils.hpp"

namespace utils
{
  double
  heaviside_sharp(const double &x, const double &x0)
  {
    return (x > x0);
  }

  double
  heaviside(const double &x, const double &x0, const double &k)
  {
    return (0.5 * (1 + std::tanh(k * (x - x0))));
  }
} // namespace utils
