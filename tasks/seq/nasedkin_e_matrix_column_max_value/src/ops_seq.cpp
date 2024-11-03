// Copyright 2023 Nasedkin Egor
#include "seq/nasedkin_e_matrix_column_max_value/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> nasedkin_e_matrix_column_max_value_seq::getRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init value for output
  res_ = std::vector<int>(taskData->outputs_count[0], 0);
  return true;
}

bool nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == taskData->inputs_count[0] / taskData->inputs_count[1];
}

bool nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxSequential::run() {
  internal_order_test();
  int rows = taskData->inputs_count[0] / taskData->inputs_count[1];
  int cols = taskData->inputs_count[1];
  for (int col = 0; col < cols; col++) {
    int max_val = input_[col];
    for (int row = 1; row < rows; row++) {
      max_val = std::max(max_val, input_[row * cols + col]);
    }
    res_[col] = max_val;
  }
  return true;
}

bool nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxSequential::post_processing() {
  internal_order_test();
  for (unsigned i = 0; i < res_.size(); i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
  }
  return true;
}