
#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vershinina_a_image_smoothing {

std::vector<int> getRandomVector(int rows, int cols);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  int rows{};
  int cols{};

 private:
  std::vector<int> output_;
  int *input_{};
};

}  // namespace vershinina_a_image_smoothing