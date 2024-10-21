// Copyright 2024 Sotskov Andrey
#include "seq/sotskov_a_sum_element_matrix/include/ops_seq.hpp"

bool sotskov_a_sum_element_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  matrix_ = *reinterpret_cast<std::vector<std::vector<int>>*>(taskData->inputs[0]);
  result_ = 0;
  return true;
}

bool sotskov_a_sum_element_matrix_seq::TestTaskSequential::validation() {
    internal_order_test();
    if (taskData->inputs_count.size() != 1 || taskData->outputs_count.size() != 1) {
        return false;
    }
    const auto& inputMatrix = *reinterpret_cast<const std::vector<std::vector<int>>*>(taskData->inputs[0]);
    return !inputMatrix.empty() && !inputMatrix[0].empty() && taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool sotskov_a_sum_element_matrix_seq::TestTaskSequential::run() {
  internal_order_test();
  for (const auto& row : matrix_) {
    for (int elem : row) {
      result_ += elem;
    }
  }
  return true;
}

bool sotskov_a_sum_element_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = result_;
  return true;
}
