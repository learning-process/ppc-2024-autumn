#include "seq/oturin_a_max_values_by_rows_matrix/include/ops_seq.hpp"

#include <algorithm>
#include <thread>

using namespace std::chrono_literals;

bool oturin_a_max_values_by_rows_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  n = (size_t)(taskData->inputs_count[0] / m);
  m = (size_t)*taskData->inputs[1];
  input_ = std::vector<int>(taskData->inputs_count[0]);
  int *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init values for output
  res = std::vector<int>(m, 0);
  return true;
}

bool oturin_a_max_values_by_rows_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool oturin_a_max_values_by_rows_matrix_seq::TestTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < m; i++) res[i] = *std::max_element(input_.begin() + i * n, input_.begin() + (i + 1) * n);
  return true;
}

bool oturin_a_max_values_by_rows_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < m; i++) {
    reinterpret_cast<int *>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}
