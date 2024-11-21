#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_monte_carlo_integration {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_,
                                 std::function<double(double)> func = function_square)
      : Task(std::move(taskData_)), function(std::move(func)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  double a = 0.0;
  double b = 0.0;
  double epsilon = 0.0;
  int num_samples = 0;

  static double function_square(double x) { return x * x; }

 private:
  double res{};
  std::function<double(double)> function;
};

}  // namespace malyshev_v_monte_carlo_integration
