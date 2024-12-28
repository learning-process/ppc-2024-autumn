#include "seq/sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned/include/ops_seq.hpp"

#include <functional>
#include <random>
#include <thread>

using namespace std::chrono_literals;

bool sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  input_A = reinterpret_cast<int*>(taskData->inputs[0]);
  input_B = reinterpret_cast<int*>(taskData->inputs[1]);

  column_A = taskData->inputs_count[0];
  row_A = taskData->inputs_count[1];
  column_B = taskData->inputs_count[2];
  row_B = taskData->inputs_count[3];

  C.assign(column_A * row_B, 0);

  return true;
}

bool sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 1 && taskData->inputs_count[1] > 0 && taskData->inputs_count[2] > 0 &&
         taskData->inputs_count[3] > 0 && taskData->inputs_count[1] == taskData->inputs_count[2];
}

bool sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq::TestTaskSequential::run() {
  internal_order_test();

  int tmp;
  for (int i = 0; i < column_A; i++) {
    for (int j = 0; j < row_B; j++) {
      C[i * row_B + j] = 0;
      tmp = 0;
      for (int k = 0; k < row_A; k++) {
        tmp += input_A[i * row_A + k] * input_B[k * row_B + j];
        C[i * row_B + j] = tmp;
      }
    }
  }

  return true;
}

bool sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  int* ans = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(C.begin(), C.end(), ans);

  return true;
}
