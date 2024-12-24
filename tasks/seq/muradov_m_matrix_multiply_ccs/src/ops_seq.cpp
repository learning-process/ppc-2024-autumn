#include "seq/muradov_m_matrix_multiply_ccs/include/ops_seq.hpp"

bool muradov_m_matrix_multiply_ccs_seq::MatrixMultiplyCCS::validation() {
  internal_order_test();

  int val_rows_A = *reinterpret_cast<int*>(taskData->inputs[0]);
  int val_cols_A = *reinterpret_cast<int*>(taskData->inputs[1]);
  int val_rows_B = *reinterpret_cast<int*>(taskData->inputs[2]);
  int val_cols_B = *reinterpret_cast<int*>(taskData->inputs[3]);

  return val_rows_A > 0 && val_cols_A > 0 && val_rows_B > 0 && val_cols_B > 0 && val_cols_A == val_rows_B;
}

bool muradov_m_matrix_multiply_ccs_seq::MatrixMultiplyCCS::pre_processing() {
  internal_order_test();

  rows_A = *reinterpret_cast<int*>(taskData->inputs[0]);
  cols_A = *reinterpret_cast<int*>(taskData->inputs[1]);
  rows_B = *reinterpret_cast<int*>(taskData->inputs[2]);
  cols_B = *reinterpret_cast<int*>(taskData->inputs[3]);

  auto* A_val_ptr = reinterpret_cast<double*>(taskData->inputs[4]);
  int A_val_size = taskData->inputs_count[4];
  A_val.assign(A_val_ptr, A_val_ptr + A_val_size);

  auto* A_row_ind_ptr = reinterpret_cast<int*>(taskData->inputs[5]);
  int A_row_ind_size = taskData->inputs_count[5];
  A_row_ind.assign(A_row_ind_ptr, A_row_ind_ptr + A_row_ind_size);

  auto* A_col_ptr_ptr = reinterpret_cast<int*>(taskData->inputs[6]);
  int A_col_ptr_size = taskData->inputs_count[6];
  A_col_ptr.assign(A_col_ptr_ptr, A_col_ptr_ptr + A_col_ptr_size);

  auto* B_val_ptr = reinterpret_cast<double*>(taskData->inputs[7]);
  int B_val_size = taskData->inputs_count[7];
  B_val.assign(B_val_ptr, B_val_ptr + B_val_size);

  auto* B_row_ind_ptr = reinterpret_cast<int*>(taskData->inputs[8]);
  int B_row_ind_size = taskData->inputs_count[8];
  B_row_ind.assign(B_row_ind_ptr, B_row_ind_ptr + B_row_ind_size);

  auto* B_col_ptr_ptr = reinterpret_cast<int*>(taskData->inputs[9]);
  int B_col_ptr_size = taskData->inputs_count[9];
  B_col_ptr.assign(B_col_ptr_ptr, B_col_ptr_ptr + B_col_ptr_size);

  transpose_CCS(A_val, A_row_ind, A_col_ptr, rows_A, cols_A, At_val, At_row_ind, At_col_ptr);

  rows_At = cols_A;
  cols_At = rows_A;

  return true;
}

bool muradov_m_matrix_multiply_ccs_seq::MatrixMultiplyCCS::run() {
  internal_order_test();

  multiply_CCS(At_val, At_row_ind, At_col_ptr, rows_At, B_val, B_row_ind, B_col_ptr, cols_B, res_val, res_ind, res_ptr);

  return true;
}

bool muradov_m_matrix_multiply_ccs_seq::MatrixMultiplyCCS::post_processing() {
  internal_order_test();

  auto* C_val_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  auto* C_row_ind_ptr = reinterpret_cast<int*>(taskData->outputs[1]);
  auto* C_col_ptr_ptr = reinterpret_cast<int*>(taskData->outputs[2]);

  std::copy(res_val.begin(), res_val.end(), C_val_ptr);
  std::copy(res_ind.begin(), res_ind.end(), C_row_ind_ptr);
  std::copy(res_ptr.begin(), res_ptr.end(), C_col_ptr_ptr);

  return true;
}
