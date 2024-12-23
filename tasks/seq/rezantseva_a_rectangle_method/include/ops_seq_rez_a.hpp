// seq hpp rezantseva_a_rectangle_method
#pragma once

#include <cmath>
#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

#define func double(const std::vector<double> &)

namespace rezantseva_a_rectangle_method_seq {
class RectangleMethodSequential : public ppc::core::Task {
 public:
  explicit RectangleMethodSequential(std::shared_ptr<ppc::core::TaskData> taskData_, std::function<func> f)
      : Task(std::move(taskData_)), func_(std::move(f)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double result_{};
  std::vector<std::pair<double, double>> integration_bounds_;
  std::vector<int> distribution_;
  std::function<func> func_;

  static bool check_integration_bounds(std::vector<std::pair<double, double>> *ib);
};
}  // namespace rezantseva_a_rectangle_method_seq
