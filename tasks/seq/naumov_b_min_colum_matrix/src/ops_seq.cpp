// Copyright 2024 Nesterov Alexander
#include "seq/naumov_b_min_colum_matrix/include/ops_seq.hpp"

bool naumov_b_min_colum_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  int st = taskData->inputs_count[0];
  int cl = taskData->inputs_count[1];

  input_.resize(st, std::vector<int>(cl));

  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  for (int i = 0; i < st; ++i) {
    for (int j = 0; j < cl; ++j) {
      input_[i][j] = input_data[i * cl + j];
    }
  }
  res.resize(cl);

  return true;
}

bool naumov_b_min_colum_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();

  return (taskData->inputs_count.size() >= 2) && (!taskData->inputs.empty()) && (!taskData->outputs.empty());
}

bool naumov_b_min_colum_matrix_seq::TestTaskSequential::run() {
  internal_order_test();

  size_t numRows = input_.size();
  size_t numCols = input_[0].size();

  for (size_t j = 0; j < numCols; j++) {
    res[j] = input_[0][j];
    for (size_t i = 1; i < numRows; i++) {
      if (input_[i][j] < res[j]) {
        res[j] = input_[i][j];
      }
    }
  }

  return true;
}

bool naumov_b_min_colum_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  int* output_ = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res.size(); i++) {
    output_[i] = res[i];
  }
  return true;
}
