#include "seq/savchenko_m_ribbon_mult_split_a/include/ops_seq_savchenko.hpp"

bool savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential::validation() {
  internal_order_test();
  // columns_A = taskData->inputs_count[0];
  // rows_A = taskData->inputs_count[1];
  // columns_B = taskData->inputs_count[2];
  // rows_B = taskData->inputs_count[3];

  bool valid_inputs = (taskData->inputs.size() == 2);
  bool valid_outputs = (taskData->outputs.size() == 1);
  bool valid_icount = (taskData->inputs_count.size() == 4);
  bool valid_ocount = (taskData->outputs_count.size() == 1);
  bool valid_io = (valid_inputs && valid_outputs && valid_icount && valid_ocount);
  if (!valid_io) return false;

  bool matrix_A_positive_size = (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0);
  bool matrix_B_positive_size = (taskData->inputs_count[2] > 0 && taskData->inputs_count[3] > 0);
  bool equal_columnsA_rowsB = (taskData->inputs_count[0] == taskData->inputs_count[3]);

  bool valid = (matrix_A_positive_size && matrix_B_positive_size && equal_columnsA_rowsB);
  return valid;
}

bool savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init values for input and output
  columns_A = taskData->inputs_count[0];
  rows_A = taskData->inputs_count[1];
  columns_B = taskData->inputs_count[2];
  rows_B = taskData->inputs_count[3];

  matrix_A = std::vector<int>(columns_A * rows_A, 0);
  matrix_B = std::vector<int>(columns_B * rows_B, 0);
  matrix_res = std::vector<int>(rows_A * columns_B, 0);

  auto *tmp_A = reinterpret_cast<int *>(taskData->inputs[0]);
  std::copy(tmp_A, tmp_A + rows_A * columns_A, matrix_A.begin());

  auto *tmp_B = reinterpret_cast<int *>(taskData->inputs[1]);
  std::copy(tmp_B, tmp_B + rows_B * columns_B, matrix_B.begin());

  return true;
}

bool savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < rows_A; i++) {
    for (size_t j = 0; j < columns_B; j++) {
      for (size_t k = 0; k < columns_A; k++) {
        matrix_res[i * columns_B + j] += matrix_A[i * columns_A + k] * matrix_B[k * columns_B + j];
      }
    }
  }
  return true;
}

bool savchenko_m_ribbon_mult_split_a_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  int *ptr_matrix_res = reinterpret_cast<int *>(taskData->outputs[0]);
  std::copy(matrix_res.begin(), matrix_res.begin() + rows_A * columns_B, ptr_matrix_res);
  return true;
}
