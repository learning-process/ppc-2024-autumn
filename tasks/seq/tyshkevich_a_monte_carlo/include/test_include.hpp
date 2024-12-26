#pragma once

#include <cmath>
#include <vector>

namespace tyshkevich_a_monte_carlo_seq {

inline double function_sin_sum(const std::vector<double>& x) {
  double sum = 0.0;
  for (double xi : x) {
    sum += std::sin(xi);
  }
  return sum;
}

inline double function_cos_product(const std::vector<double>& x) {
  double product = 1.0;
  for (double xi : x) {
    product *= std::cos(xi);
  }
  return product;
}

inline double function_gaussian(const std::vector<double>& x) {
  double sum = 0.0;
  for (double xi : x) {
    sum += xi * xi;
  }
  return std::exp(-sum);
}

inline double function_paraboloid(const std::vector<double>& x) {
  double sum = 0.0;
  for (double xi : x) {
    sum += xi * xi;
  }
  return sum;
}

inline double function_exp_sum(const std::vector<double>& x) {
  double sum = 0.0;
  for (double xi : x) {
    sum += std::exp(xi);
  }
  return sum;
}

inline double function_abs_product(const std::vector<double>& x) {
  double product = 1.0;
  for (double xi : x) {
    product *= std::abs(xi);
  }
  return product;
}

inline double function_log_sum_squares(const std::vector<double>& x) {
  double sum = 0.0;
  for (double xi : x) {
    sum += xi * xi;
  }
  return std::log(1 + sum);
}

}  // namespace tyshkevich_a_monte_carlo_seq
