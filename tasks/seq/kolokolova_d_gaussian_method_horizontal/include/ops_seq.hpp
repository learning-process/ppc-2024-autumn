// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace kolokolova_d_gaussian_method_horizontal_seq {

std::vector<int> getRandomVector(int sz);
int find_rank(std::vector<double>& matrix, int rows, int cols);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_coeff;
  std::vector<int> input_y;
  std::vector<double> res;
  int count_equations = 0;
};

}  // namespace kolokolova_d_gaussian_method_horizontal_seq