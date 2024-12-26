// Copyright 2024 Korobeinikov Brseny
#include "seq/korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B/include/ops_seq_korobeinikov.hpp"

using namespace std::chrono_literals;

bool korobeinikov_a_test_task_seq_lab_02::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  A.data.reserve(taskData->inputs_count[0]);
  auto *tmp_ptr_1 = reinterpret_cast<int *>(taskData->inputs[0]);
  std::copy(tmp_ptr_1, tmp_ptr_1 + taskData->inputs_count[0], A.data.begin());
  A.count_rows = (int)*taskData->inputs[1];
  A.count_cols = (int)*taskData->inputs[2];

  B.data.reserve(taskData->inputs_count[3]);
  auto *tmp_ptr_2 = reinterpret_cast<int *>(taskData->inputs[3]);
  std::copy(tmp_ptr_2, tmp_ptr_2 + taskData->inputs_count[3], B.data.begin());
  B.count_rows = (int)*taskData->inputs[4];
  B.count_cols = (int)*taskData->inputs[5];

  res = Matrix(A.count_rows, B.count_cols);
  return true;
}

bool korobeinikov_a_test_task_seq_lab_02::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs.size() == 6 && taskData->inputs_count.size() == 6 && taskData->outputs.size() == 3 &&
         taskData->outputs_count.size() == 3 && (*taskData->inputs[2] == *taskData->inputs[4]) &&
         ((*taskData->inputs[1]) * (*taskData->inputs[2]) == (int)taskData->inputs_count[0]) &&
         ((*taskData->inputs[4]) * (*taskData->inputs[5]) == (int)taskData->inputs_count[3]);
}

bool korobeinikov_a_test_task_seq_lab_02::TestTaskSequential::run() {
  internal_order_test();

  for (int i = 0; i < A.count_rows; i++) {
    for (int j = 0; j < B.count_cols; j++) {
      res.get_el(i, j) = 0;
      for (int k = 0; k < A.count_cols; k++) {
        res.get_el(i, j) += A.get_el(i, k) * B.get_el(k, j);
      }
    }
  }

  return true;
}

bool korobeinikov_a_test_task_seq_lab_02::TestTaskSequential::post_processing() {
  internal_order_test();

  std::copy(res.data.begin(), res.data.end(), reinterpret_cast<int *>(taskData->outputs[0]));
  *reinterpret_cast<int *>(taskData->outputs[1]) = res.count_rows;
  *reinterpret_cast<int *>(taskData->outputs[2]) = res.count_cols;
  return true;
}
