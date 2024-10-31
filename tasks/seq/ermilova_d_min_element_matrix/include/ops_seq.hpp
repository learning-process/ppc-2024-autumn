// Copyright 2023 Nesterov Alexander
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace ermilova_d_min_element_matrix_seq {
std::vector<int> getRandomVector(int size, int upper_border, int lower_border);
std::vector<std::vector<int>> getRandomMatrix(int rows, int cols, int upper_border, int lower_border);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> input_;
  int res{};
  unsigned int cols, rows;
};

}  // namespace ermilova_d_min_element_matrix_seq