// Copyright 2024 Nesterov Alexander
#include "seq/beresnev_a_min_values_by_matrix_columns/include/ops_seq.hpp"

#include <limits>
#include <thread>

using namespace std::chrono_literals;

bool beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = reinterpret_cast<std::vector<int>*>(taskData->inputs[0])[0];
  res_ = reinterpret_cast<std::vector<int>*>(taskData->outputs[0])[0];
  n_ = reinterpret_cast<int*>(taskData->inputs[1])[0];
  m_ = reinterpret_cast<int*>(taskData->inputs[2])[0];
  return true;
}

bool beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 &&
         taskData->inputs_count[0] == reinterpret_cast<std::vector<int>*>(taskData->inputs[0])[0].size() &&
         taskData->inputs_count[0] == (uint32_t) reinterpret_cast<int*>(taskData->inputs[1])[0] *
                                          reinterpret_cast<int*>(taskData->inputs[2])[0] &&
         taskData->outputs_count[0] == reinterpret_cast<std::vector<int>*>(taskData->outputs[0])[0].size();
}

bool beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < m_; i++) {
    int min = input_[i];
    for (int j = 1; j < n_; j++) {
      if (input_[j * m_ + i] < min) {
        min = input_[j * m_ + i];
      }
    }
    res_[i] = min;
  }
  return true;
}

bool beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<int>*>(taskData->outputs[0])[0] = res_;
  return true;
}
