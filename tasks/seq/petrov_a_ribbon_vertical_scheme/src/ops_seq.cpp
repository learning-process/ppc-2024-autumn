// Copyright 2024 Nesterov Alexander
#include "seq/petrov_a_ribbon_vertical_scheme/include/ops_seq.hpp"

#include <stdexcept>

namespace petrov_a_ribbon_vertical_scheme_seq {

bool TestTaskSequential::pre_processing() {
  internal_order_test();

  int rows = taskData->inputs_count[0];
  int cols = taskData->inputs_count[1];
  matrix_.resize(rows, std::vector<int>(cols));
  vector_.resize(cols);
  result_.resize(rows, 0);

  int* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_[i][j] = matrix_data[i * cols + j];
    }
  }

  int* vector_data = reinterpret_cast<int*>(taskData->inputs[1]);
  for (int i = 0; i < cols; ++i) {
    vector_[i] = vector_data[i];
  }

  return true;
}

bool TestTaskSequential::validation() {
  internal_order_test();

  bool isValid = (!taskData->inputs_count.empty() && !taskData->inputs.empty() && !taskData->outputs.empty());
  return isValid;
}

bool TestTaskSequential::run() {
  internal_order_test();

  int rows = matrix_.size();
  int cols = vector_.size();

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result_[i] += matrix_[i][j] * vector_[j];
    }
  }

  return true;
}

bool TestTaskSequential::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < result_.size(); ++i) {
    output_data[i] = result_[i];
  }

  return true;
}

}  // namespace petrov_a_ribbon_vertical_scheme_seq
