#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace anufriev_d_linear_image {

class SimpleIntSEQ : public ppc::core::Task {
 public:
  explicit SimpleIntSEQ(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  void applyGaussianFilter();

  std::vector<int> input_data_;
  std::vector<int> processed_data_;
  int rows;
  int cols;

  const int kernel_[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
};

}  // namespace anufriev_d_linear_image