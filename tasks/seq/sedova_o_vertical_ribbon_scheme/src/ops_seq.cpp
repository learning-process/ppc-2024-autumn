#include "seq/sedova_o_vertical_ribbon_scheme/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

bool sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 1 &&
         taskData->inputs_count[0] / taskData->inputs_count[1] > 0;  // Check for valid matrix dimension
}

bool sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  int* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];

  int* vector_data = reinterpret_cast<int*>(taskData->inputs[1]);
  int vector_size = taskData->inputs_count[1];

  input_matrix_.assign(matrix_data, matrix_data + matrix_size);
  input_vector_.assign(vector_data, vector_data + vector_size);

  num_cols_ = input_vector_.size();
  num_rows_ = input_matrix_.size() / num_cols_;

  int result_size = taskData->outputs_count[0];
  result_vector_.resize(result_size, 0);

  return true;
}

bool sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential::run() {
  internal_order_test();

  for (int j = 0; j < num_cols_; ++j) {
    for (int i = 0; i < num_rows_; ++i) {
      result_vector_[i] += input_matrix_[i * num_cols_ + j] * input_vector_[j];
    }
  }
  return true;
}

bool sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_vector_.begin(), result_vector_.end(), output_data);

  return true;
}