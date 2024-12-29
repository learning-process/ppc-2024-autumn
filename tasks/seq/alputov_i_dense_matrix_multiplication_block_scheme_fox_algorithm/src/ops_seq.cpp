#include "seq/alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm/include/ops_seq.hpp"

#include <functional>

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq::pre_processing() {
  internal_order_test();

  auto* input_A = reinterpret_cast<double*>(taskData->inputs[0]);
  row_A = static_cast<int>(taskData->inputs_count[0]);
  column_A = static_cast<int>(taskData->inputs_count[1]);

  auto* input_B = reinterpret_cast<double*>(taskData->inputs[1]);
  row_B = static_cast<int>(taskData->inputs_count[2]);
  column_B = static_cast<int>(taskData->inputs_count[3]);

  A.resize(column_A * row_A);
  B.resize(column_B * row_B);

  std::copy(input_A, input_A + column_A * row_A, A.begin());
  std::copy(input_B, input_B + column_B * row_B, B.begin());

  C.resize(row_A * column_B, 0.0);

  return true;
}

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq::validation() {
  internal_order_test();

  return static_cast<int>(taskData->inputs_count[0]) > 0 && static_cast<int>(taskData->inputs_count[1]) > 0 &&
         static_cast<int>(taskData->inputs_count[2]) > 0 && static_cast<int>(taskData->inputs_count[3]) > 0 &&
         static_cast<int>(taskData->inputs_count[1]) == static_cast<int>(taskData->inputs_count[2]);
}

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq::run() {
  internal_order_test();
  for (int i = 0; i < row_A; ++i) {
    for (int j = 0; j < column_B; ++j) {
      for (int k = 0; k < column_A; ++k) {
        C[i * column_B + j] += A[i * column_A + k] * B[k * column_B + j];
      }
    }
  }

  return true;
}

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq::post_processing() {
  internal_order_test();

  auto* res = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(C.begin(), C.end(), res);
  return true;
}