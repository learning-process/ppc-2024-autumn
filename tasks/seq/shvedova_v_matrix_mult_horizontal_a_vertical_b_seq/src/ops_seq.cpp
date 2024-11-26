#include "seq/shvedova_v_matrix_mult_horizontal_a_vertical_b_seq/include/ops_seq.hpp"

#include <vector>

bool shvedova_v_matrix_mult_horizontal_a_vertical_b_seq::MatrixMultiplicationTaskSequential::pre_processing() {
  internal_order_test();

  auto* sizes = reinterpret_cast<size_t*>(taskData->inputs[0]);
  row_a = sizes[0];
  col_a = sizes[1];
  row_b = sizes[2];
  col_b = sizes[3];

  matrix_a.resize(row_a * col_a);
  matrix_b.resize(row_b * col_b);
  matrix_c.resize(row_a * col_b, 0);

  int* matrix_a_ptr = reinterpret_cast<int*>(taskData->inputs[1]);
  for (size_t i = 0; i < row_a; ++i) {
    for (size_t j = 0; j < col_a; ++j) {
      matrix_a[i * col_a + j] = matrix_a_ptr[i * col_a + j];
    }
  }

  int* matrix_b_ptr = reinterpret_cast<int*>(taskData->inputs[2]);
  for (size_t i = 0; i < row_b; ++i) {
    for (size_t j = 0; j < col_b; ++j) {
      matrix_b[i * col_b + j] = matrix_b_ptr[i * col_b + j];
    }
  }

  return true;
}

bool shvedova_v_matrix_mult_horizontal_a_vertical_b_seq::MatrixMultiplicationTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count.size() < 3 || taskData->outputs_count.empty()) {
    return false;
  }

  auto* sizes = reinterpret_cast<size_t*>(taskData->inputs[0]);

  return (sizes[1] == sizes[2]) &&
         (taskData->inputs_count[1] == sizes[0] * sizes[1] && taskData->inputs_count[2] == sizes[2] * sizes[3] &&
          taskData->outputs_count[0] == sizes[0] * sizes[3] && (sizes[0] * sizes[1] * sizes[2] * sizes[3] != 0));
}

bool shvedova_v_matrix_mult_horizontal_a_vertical_b_seq::MatrixMultiplicationTaskSequential::run() {
  internal_order_test();

  for (size_t i = 0; i < row_a; ++i) {
    for (size_t j = 0; j < col_b; ++j) {
      int sum = 0;
      for (size_t k = 0; k < col_a; ++k) {
        sum += matrix_a[i * col_a + k] * matrix_b[k * col_b + j];
      }
      matrix_c[i * col_b + j] = sum;
    }
  }

  return true;
}

bool shvedova_v_matrix_mult_horizontal_a_vertical_b_seq::MatrixMultiplicationTaskSequential::post_processing() {
  internal_order_test();

  for (size_t i = 0; i < row_a * col_b; ++i) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = matrix_c[i];
  }

  return true;
}