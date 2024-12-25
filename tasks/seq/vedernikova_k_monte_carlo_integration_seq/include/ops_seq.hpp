#pragma once

#include <cstddef>
#include <functional>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace vedernikova_k_monte_carlo_integration_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::function<double(double, double)> f;

 private:
  double a_x, b_x, a_y, b_y;
  int num_points;
  double res{};
};

}  // namespace vedernikova_k_monte_carlo_integration_seq