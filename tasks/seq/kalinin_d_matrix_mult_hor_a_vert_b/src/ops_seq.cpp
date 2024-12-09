#include "seq/kalinin_d_matrix_mult_hor_a_vert_b/include/ops_seq.hpp"

#include <algorithm>
#include <thread>

using namespace std::chrono_literals;

bool kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential::pre_processing() {
  internal_order_test();

  input_A = reinterpret_cast<int*>(taskData->inputs[0]);
  input_B = reinterpret_cast<int*>(taskData->inputs[1]);

  rows_A = taskData->inputs_count[0];
  columns_A = taskData->inputs_count[1];
  rows_B = taskData->inputs_count[2];
  columns_B = taskData->inputs_count[3];

  if (columns_A != rows_B) {
    return false;
  }

  C.assign(rows_A * columns_B, 0);

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential::validation() {
  internal_order_test();

  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->inputs_count[2] > 0 &&
         taskData->inputs_count[3] > 0 && taskData->inputs_count[1] == taskData->inputs_count[2];
}

bool kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential::run() {
  internal_order_test();

  for (int i = 0; i < rows_A; ++i) {
    for (int j = 0; j < columns_B; ++j) {
      for (int k = 0; k < columns_A; ++k) {
        C[i * columns_B + j] += input_A[i * columns_A + k] * input_B[k * columns_B + j];
      }
    }
  }

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential::post_processing() {
  internal_order_test();

  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);

  std::copy(C.begin(), C.end(), output_data);

  return true;
}
