// Copyright 2024 Korobeinikov Arseny
#include "seq/korobeinikov_a_max_elements_in_rows_of_matrix/include/ops_seq_korobeinikov.hpp"

#include <algorithm>
#include <thread>

using namespace std::chrono_literals;

bool korobeinikov_a_test_task_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output

  input_.reserve(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], std::back_inserter(input_));
  count_rows = (int)*taskData->inputs[1];
  if (count_rows != 0) {
    size_rows = (int)(taskData->inputs_count[0] / (*taskData->inputs[1]));
  } else {
    size_rows = 0;
  }

  res = std::vector<int>(count_rows, 0);
  return true;
}

bool korobeinikov_a_test_task_seq::TestTaskSequential::validation() {
  internal_order_test();

  if ((*taskData->inputs[1]) == 0) {
    return true;
  } else {
    return (*taskData->inputs[1] == taskData->outputs_count[0] &&
            (taskData->inputs_count[0] % (*taskData->inputs[1])) == 0);
  }
}

bool korobeinikov_a_test_task_seq::TestTaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < count_rows; i++) {
    res[i] = *std::max_element(input_.begin() + i * size_rows, input_.begin() + (i + 1) * size_rows);
  }
  return true;
}

bool korobeinikov_a_test_task_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < count_rows; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}
