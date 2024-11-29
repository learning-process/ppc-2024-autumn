#include "seq/sedova_o_vertical_ribbon_scheme/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

bool sedova_o_vertical_ribbon_scheme_seq::Sequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 1 && taskData->inputs_count[1] >= 1 && !taskData &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0 && taskData->outputs[0] != nullptr;
}

bool sedova_o_vertical_ribbon_scheme_seq::Sequential::pre_processing() {
  internal_order_test();

  input_matrix_ = reinterpret_cast<int*>(taskData->inputs[0]);
  input_vector_ = reinterpret_cast<int*>(taskData->inputs[1]);
  int count = taskData->inputs_count[0];
  rows_ = taskData->inputs_count[1];
  cols_ = count / rows_;
  result_vector_.assign(cols_, 0);

  return true;
}

bool sedova_o_vertical_ribbon_scheme_seq::Sequential::run() {
  internal_order_test();

  for (int j = 0; j < cols_; ++j) {
    for (int i = 0; i < rows_; ++i) {
      result_vector_[i] += input_matrix_[i * cols_ + j] * input_vector_[j];
    }
  }
  return true;
}

bool sedova_o_vertical_ribbon_scheme_seq::Sequential::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_vector_.begin(), result_vector_.end(), output_data);

  return true;
}