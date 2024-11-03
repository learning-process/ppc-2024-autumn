#include "seq/nasedkin_e_matrix_column_max_value/include/ops_seq.hpp"

#include <algorithm>
#include <numeric>

bool nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxTaskSequential::pre_processing() {
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  res = 0;
  return true;
}

bool nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxTaskSequential::validation() {
  return taskData->outputs_count[0] == 1;
}

bool nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxTaskSequential::run() {
  res = *std::max_element(input_.begin(), input_.end());
  return true;
}

bool nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxTaskSequential::post_processing() {
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
