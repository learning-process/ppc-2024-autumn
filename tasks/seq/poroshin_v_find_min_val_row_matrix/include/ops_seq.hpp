// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace poroshin_v_find_min_val_row_matrix_seq {

  std::vector<int> gen(int m, int n);  // generate vector (matrix)

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_{}, res{};
  //inputs = vector (matrix)
  //inputs_count[0] = m, inputs_count[1] = n 
  // m - num of rows, n - num of columns
};

}  // namespace poroshin_v_find_min_val_row_matrix_seq