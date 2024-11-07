#pragma once
#include <functional>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace vershinina_a_integration_the_monte_carlo_method {

std::vector<float> getRandomVector(float sz);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::function<float(float)> p;

 private:
  float xmin{};
  float xmax{};
  float ymin{};
  float ymax{};
  float *input_{};
  float reference_res{};
};
}  // namespace vershinina_a_integration_the_monte_carlo_method