// Copyright 2024 Nesterov Alexander
#include "seq/korovin_n_matrix_multiple_cannon/include/ops_seq.hpp"

#include <thread>

bool korovin_n_matrix_multiple_cannon_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  m = taskData->inputs_count[0];
  n = taskData->inputs_count[1];
  k = taskData->inputs_count[2];
  auto* a_data = reinterpret_cast<double*>(taskData->inputs[0]);
  A_.assign(a_data, a_data + (m * n));
  auto* b_data = reinterpret_cast<double*>(taskData->inputs[1]);
  B_.assign(b_data, b_data + (n * k));

  return true;
}

bool korovin_n_matrix_multiple_cannon_seq::TestTaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 3) return false;

  m = taskData->inputs_count[0];
  n = taskData->inputs_count[1];
  k = taskData->inputs_count[2];

  return (m > 0 && n > 0 && k > 0) &&
         (taskData->inputs.size() >= 2 && taskData->inputs[0] != nullptr && taskData->inputs[1] != nullptr) &&
         (!taskData->outputs.empty() && taskData->outputs[0] != nullptr);
}

bool korovin_n_matrix_multiple_cannon_seq::TestTaskSequential::run() {
  internal_order_test();

  C_.resize(m * k, 0.0);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      for (int p = 0; p < n; p++) {
        C_[i * k + j] += A_[i * n + p] * B_[p * k + j];
      }
    }
  }

  return true;
}

bool korovin_n_matrix_multiple_cannon_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  auto* data_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(C_.begin(), C_.end(), data_ptr);

  return true;
}
