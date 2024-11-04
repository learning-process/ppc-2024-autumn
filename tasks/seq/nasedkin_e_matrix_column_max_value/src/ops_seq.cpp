// Copyright 2023 Nasedkin Egor
#include "seq/nasedkin_e_matrix_column_max_value/include/ops_seq.hpp"

#include <algorithm>
#include <random>
#include <vector>

namespace nasedkin_e_matrix_column_max_value_seq {

std::vector<int> getRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool MatrixColumnMaxSeq::pre_processing() {
  internal_order_test();
  int rows = taskData->inputs_count[0] / taskData->inputs_count[1];
  int cols = taskData->inputs_count[1];
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }

  res_ = std::vector<int>(cols, 0);
  return true;
}

bool MatrixColumnMaxSeq::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == taskData->inputs_count[1];
}

bool MatrixColumnMaxSeq::run() {
  internal_order_test();
  int rows = input_.size() / res_.size();
  for (int col = 0; col < res_.size(); col++) {
    res_[col] = input_[col];
    for (int row = 1; row < rows; row++) {
      if (input_[row * res_.size() + col] > res_[col]) {
        res_[col] = input_[row * res_.size() + col];
      }
    }
  }
  return true;
}

bool MatrixColumnMaxSeq::post_processing() {
  internal_order_test();
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  for (unsigned i = 0; i < res_.size(); i++) {
    tmp_ptr[i] = res_[i];
  }
  return true;
}

}  // namespace nasedkin_e_matrix_column_max_value_seq