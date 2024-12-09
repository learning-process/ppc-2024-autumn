#pragma once
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
namespace lysov_i_integration_the_trapezoid_method_seq {
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  double a, b, cnt_of_splits, epsilon, h;
  static double function_square(double x) { return x * x; }

 private:
  std::vector<double> input_;
  double res{};
};
}  // namespace lysov_i_integration_the_trapezoid_method_seq