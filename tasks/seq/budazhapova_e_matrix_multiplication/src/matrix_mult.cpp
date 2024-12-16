#include "seq/budazhapova_e_matrix_multiplication/include/matrix_mult.hpp"

#include <thread>

bool budazhapova_e_matrix_mult_seq::MatrixMultSequential::pre_processing() {
  internal_order_test();

  A = std::vector<int>(reinterpret_cast<int*>(taskData->inputs[0]),
                       reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0]);
  b = std::vector<int>(reinterpret_cast<int*>(taskData->inputs[1]),
                       reinterpret_cast<int*>(taskData->inputs[1]) + taskData->inputs_count[1]);
  columns = taskData->inputs_count[1];
  rows = taskData->inputs_count[0] / columns;
  res = std::vector<int>(rows);
  return true;
}

bool budazhapova_e_matrix_mult_seq::MatrixMultSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[1] > 0 && taskData->inputs_count[0] % taskData->inputs_count[1] == 0 &&
         taskData->inputs_count[0] > 0;
}

bool budazhapova_e_matrix_mult_seq::MatrixMultSequential::run() {
  internal_order_test();
  for (int i = 0; i < rows; i++) {
    res[i] = 0;
    for (int j = 0; j < columns; j++) {
      res[i] += A[j + columns * i] * b[j];
    }
  }
  return true;
}

bool budazhapova_e_matrix_mult_seq::MatrixMultSequential::post_processing() {
  internal_order_test();
  int* output = reinterpret_cast<int*>(taskData->outputs[0]);
  for (int i = 0; i < rows; i++) {
    output[i] = res[i];
  }

  return true;
}
