#pragma once

#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

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

class MonteCarloSequential : public ppc::core::Task {
 public:
  explicit MonteCarloSequential(std::shared_ptr<ppc::core::TaskData> taskData_,
                                std::function<double(const std::vector<double>&)> func_)
      : Task(std::move(taskData_)), func(std::move(func_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::function<double(const std::vector<double>&)> func;
  int dimensions;
  double precision, globalSum, result = 0.0;

  std::vector<std::pair<double, double>> bounds;
  std::mt19937 gen;
  std::vector<std::uniform_real_distribution<double>> distributions;
};

}  // namespace tyshkevich_a_monte_carlo_seq