// Copyright 2023 Nesterov Alexander
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace yasakova_t_min_of_vector_elements_seq {
std::vector<int> RandomVector(int size, int minimum = 0, int maximum = 100);
std::vector<std::vector<int>> RandomMatrix(int rows, int columns, int minimum = 0, int maximum = 100);
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_;
  int res_{};
};
}  // namespace yasakova_t_min_of_vector_elements_seq
