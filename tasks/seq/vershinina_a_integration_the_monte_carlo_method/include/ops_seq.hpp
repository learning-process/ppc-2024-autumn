#pragma once
#include <functional>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace vershinina_a_integration_the_monte_carlo_method {
std::vector<double> getRandomVector();
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::function<double(double)> p;

 private:
  double xmin{};
  double xmax{};
  double ymin{};
  double ymax{};
  double *input_{};
  double iter_count{};
  double reference_res{};
};
}  // namespace vershinina_a_integration_the_monte_carlo_method