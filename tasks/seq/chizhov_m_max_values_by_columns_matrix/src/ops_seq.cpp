// Copyright 2024 Nesterov Alexander
#include "seq/chizhov_m_max_values_by_columns_matrix/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <thread>

using namespace std::chrono_literals;

bool chizhov_m_max_values_by_columns_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  cols = (int)*taskData->inputs[1];
  rows = (int)(taskData->inputs_count[0] / cols);
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);

  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }

  res_ = std::vector<int>(cols, 0);

  return true;
}

bool chizhov_m_max_values_by_columns_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();
  if ((int)*taskData->inputs[1] == 0) {
    return false;
  }
  if (taskData->inputs.empty() || taskData->inputs_count[0] <= 0) {
    return false;
  }
  if (*taskData->inputs[1] != taskData->outputs_count[0]) {
    return false;
  }
  return true;
}

bool chizhov_m_max_values_by_columns_matrix_seq::TestTaskSequential::run() {
  internal_order_test();

  for (int j = 0; j < cols; j++) {
    int maxElement = input_[j];
    for (int i = 1; i < rows; i++) {
      if (input_[i * cols + j] > maxElement) {
        maxElement = input_[i * cols + j];
      }
    }
    res_[j] = maxElement;
  }

  return true;
}

bool chizhov_m_max_values_by_columns_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  for (int j = 0; j < cols; j++) {
    reinterpret_cast<int*>(taskData->outputs[0])[j] = res_[j];
  }

  return true;
}