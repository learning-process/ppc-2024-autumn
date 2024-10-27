// Copyright 2024 Korobeinikov Arseny
#include "seq/korobeinikov_a_max_elements_in_rows_of_matrix/include/ops_seq_korobeinikov.hpp"

#include <thread>

using namespace std::chrono_literals;

bool korobeinikov_a_test_task_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output

  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }

  count_rows = (int)*taskData->inputs[1];
  size_rows = (int)(taskData->inputs_count[0] / (*taskData->inputs[1]));
  res = std::vector<int>(count_rows, 0);
  return true;
}

bool korobeinikov_a_test_task_seq::TestTaskSequential::validation() {
  internal_order_test();

  bool flag = true;
  // Check count elements of output == count rows
  if (*taskData->inputs[1] != taskData->outputs_count[0]) {
    flag = false;
  }
  // Check equal number of elements in rows
  if ((taskData->inputs_count[0] % (*taskData->inputs[1])) != 0) {
    flag = false;
  }
  return flag;
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
