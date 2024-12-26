#include "seq/vasilev_s_striped_horizontal_scheme/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

bool vasilev_s_striped_horizontal_scheme_seq::StripedHorizontalSchemeSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 1;
}

bool vasilev_s_striped_horizontal_scheme_seq::StripedHorizontalSchemeSequential::pre_processing() {
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

bool vasilev_s_striped_horizontal_scheme_seq::StripedHorizontalSchemeSequential::run() {
  internal_order_test();

  for (int i = 0; i < num_rows_; i++) {
    int sum = 0;
    for (int j = 0; j < num_cols_; j++) {
      sum += input_matrix_[i * num_cols_ + j] * input_vector_[j];
    }
    result_vector_[i] = sum;
  }

  return true;
}

bool vasilev_s_striped_horizontal_scheme_seq::StripedHorizontalSchemeSequential::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_vector_.begin(), result_vector_.end(), output_data);

  return true;
}
