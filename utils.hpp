#pragma once

#include "common.hpp"

namespace utils
{
  double
  heaviside_sharp(const double &x, const double &x0);

  double
  heaviside(const double &x, const double &x0, const double &k);
} // namespace utils
