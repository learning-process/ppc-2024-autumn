#include "seq/sedova_o_vertical_ribbon_scheme/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

bool sedova_o_vertical_ribbon_scheme_seq::Sequential::validation() {
  internal_order_test();
  if (!taskData) {
    return false;
  }
  if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr) {
    return false;
  }
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->inputs_count[0] % taskData->inputs_count[1] == 0 &&
         taskData->outputs_count[0] == taskData->inputs_count[0] / taskData->inputs_count[1];
}

bool sedova_o_vertical_ribbon_scheme_seq::Sequential::pre_processing() {
  internal_order_test();

  matrix_ = reinterpret_cast<int*>(taskData->inputs[0]);
  count = taskData->inputs_count[0];
  vector_ = reinterpret_cast<int*>(taskData->inputs[1]);
  cols_ = taskData->inputs_count[1];
  rows_ = count / cols_;
  input_vector_.assign(vector_, vector_ + cols_);
  result_vector_.assign(rows_, 0);

  return true;
}

bool sedova_o_vertical_ribbon_scheme_seq::Sequential::run() {
  internal_order_test();

  for (int j = 0; j < cols_; ++j) {
    for (int i = 0; i < rows_; ++i) {
      result_vector_[i] += matrix_[i + j * rows_] * input_vector_[j];
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