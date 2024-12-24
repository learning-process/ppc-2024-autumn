#include "../include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>

bool sorochkin_d_matrix_col_min_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  rows_ = taskData->inputs_count[0];
  cols_ = taskData->inputs_count[1];

  const auto* src = reinterpret_cast<int*>(taskData->inputs[0]);
  input_.assign(src, src + (rows_ * cols_));

  res_.resize(cols_, std::numeric_limits<int>::max());

  return true;
}

bool sorochkin_d_matrix_col_min_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->outputs_count[0] == taskData->inputs_count[1];
}

bool sorochkin_d_matrix_col_min_seq::TestTaskSequential::run() {
  internal_order_test();

  for (size_t i = 0; i < cols_; i++) {
    for (size_t j = 0; j < rows_; j++) {
      res_[i] = std::min(res_[i], input_[j * cols_ + i]);
    }
  }

  return true;
}

bool sorochkin_d_matrix_col_min_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}
