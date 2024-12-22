#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <numbers>
#include <numeric>
#include <utility>
#include <vector>


#include "core/task/include/task.hpp"

namespace bessonov_e_multi_integration_trapezoid_method_seq {

class TestTaskSequential : public ppc::core::Task {
 public:TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  std::function<double(const std::vector<double>&)> integrand;

 private:
  size_t dim;
  std::vector<double> lower_bounds;
  std::vector<double> upper_bounds;
  std::vector<int> num_steps;

  double result;
};

}  // namespace bessonov_e_multi_integration_trapezoid_method_seq