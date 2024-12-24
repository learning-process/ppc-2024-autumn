#include "seq/korovin_n_matrix_multiple_cannon/include/ops_seq.hpp"

#include <thread>

bool korovin_n_matrix_multiple_cannon_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  numRowsA_ = taskData->inputs_count[0];
  numColsA_RowsB_ = taskData->inputs_count[1];
  numColsB_ = taskData->inputs_count[2];

  auto* a_data = reinterpret_cast<double*>(taskData->inputs[0]);
  A_.assign(a_data, a_data + (numRowsA_ * numColsA_RowsB_));
  auto* b_data = reinterpret_cast<double*>(taskData->inputs[1]);
  B_.assign(b_data, b_data + (numColsA_RowsB_ * numColsB_));

  return true;
}

bool korovin_n_matrix_multiple_cannon_seq::TestTaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 3) return false;

  int temp_numRowsA = taskData->inputs_count[0];
  int temp_numColsA_RowsB = taskData->inputs_count[1];
  int temp_numColsB = taskData->inputs_count[2];

  return (temp_numRowsA > 0 && temp_numColsA_RowsB > 0 && temp_numColsB > 0) &&
         (taskData->inputs.size() >= 2 && taskData->inputs[0] != nullptr && taskData->inputs[1] != nullptr) &&
         (!taskData->outputs.empty() && taskData->outputs[0] != nullptr);
}

bool korovin_n_matrix_multiple_cannon_seq::TestTaskSequential::run() {
  internal_order_test();

  C_.resize(numRowsA_ * numColsB_);
  for (int i = 0; i < numRowsA_; i++) {
    for (int j = 0; j < numColsB_; j++) {
      for (int p = 0; p < numColsA_RowsB_; p++) {
        C_[i * numColsB_ + j] += A_[i * numColsA_RowsB_ + p] * B_[p * numColsB_ + j];
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
