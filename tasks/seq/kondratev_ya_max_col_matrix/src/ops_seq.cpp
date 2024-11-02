// Copyright 2024 Nesterov Alexander
#include "seq/kondratev_ya_max_col_matrix/include/ops_seq.hpp"

#include <algorithm>
#include <thread>

using namespace std::chrono_literals;

bool kondratev_ya_max_col_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  uint32_t row = taskData->inputs_count[0];
  uint32_t col = taskData->inputs_count[1];

  std::vector<int32_t*> tmp(row);
  for (uint32_t i = 0; i < row; i++) {
    tmp[i] = reinterpret_cast<int32_t*>(taskData->inputs[i]);
  }

  input_.resize(col, std::vector<int32_t>(row));
  for (uint32_t j = 0; j < col; j++) {
    for (uint32_t i = 0; i < row; i++) {
      input_[j][i] = tmp[i][j];
    }
  }
  res_.resize(col);

  return true;
}

bool kondratev_ya_max_col_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == taskData->inputs_count[1] && !taskData->outputs.empty() &&
         !taskData->inputs.empty();
}

bool kondratev_ya_max_col_matrix_seq::TestTaskSequential::run() {
  internal_order_test();

  for (uint32_t i = 0; i < input_.size(); i++) {
    res_[i] = *std::max_element(input_[i].begin(), input_[i].end());
  }

  return true;
}

bool kondratev_ya_max_col_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  auto* output_matrix = reinterpret_cast<int32_t*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), output_matrix);

  return true;
}
