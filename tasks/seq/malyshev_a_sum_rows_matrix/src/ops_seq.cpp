#include "seq/malyshev_a_sum_rows_matrix/include/ops_seq.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

bool malyshev_a_sum_rows_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  uint32_t rows = taskData->inputs_count[0];
  uint32_t cols = taskData->inputs_count[1];

  input_.resize(rows, std::vector<int32_t>(cols));
  res_.resize(rows);

  int32_t* data;
  for (uint32_t i = 0; i < input_.size(); i++) {
    data = reinterpret_cast<int32_t*>(taskData->inputs[i]);
    std::copy(data, data + cols, input_[i].data());
  }

  return true;
}

bool malyshev_a_sum_rows_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool malyshev_a_sum_rows_matrix_seq::TestTaskSequential::run() {
  internal_order_test();

  for (uint32_t i = 0; i < input_.size(); i++) {
    res_[i] = std::accumulate(input_[i].begin(), input_[i].end(), 0);
  }

  return true;
}

bool malyshev_a_sum_rows_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  std::copy(res_.begin(), res_.end(), reinterpret_cast<int32_t*>(taskData->outputs[0]));

  return true;
}
