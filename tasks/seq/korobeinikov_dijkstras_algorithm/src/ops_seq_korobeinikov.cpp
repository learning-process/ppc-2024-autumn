// Copyright 2024 Korobeiinikov Arseny
#include "seq/korobeinikov_dijkstras_algorithm/include/ops_seq_korobeinikov.hpp"

#include <algorithm>

bool korobeinikov_a_test_task_seq_lab_03::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  values.reserve(taskData->inputs_count[0]);
  auto *tmp_ptr_1 = reinterpret_cast<int *>(taskData->inputs[0]);
  std::copy(tmp_ptr_1, tmp_ptr_1 + taskData->inputs_count[0], values.begin());

  col.reserve(taskData->inputs_count[1]);
  auto *tmp_ptr_2 = reinterpret_cast<int *>(taskData->inputs[1]);
  std::copy(tmp_ptr_2, tmp_ptr_2 + taskData->inputs_count[1], col.begin());

  RowIndex.reserve(taskData->inputs_count[2]);
  auto *tmp_ptr_3 = reinterpret_cast<int *>(taskData->inputs[2]);
  std::copy(tmp_ptr_3, tmp_ptr_3 + taskData->inputs_count[2], RowIndex.begin());

  size = *reinterpret_cast<int *>(taskData->inputs[3]);
  sv = *reinterpret_cast<int *>(taskData->inputs[4]);

  res = std::vector<int>(size, 0);
  return true;
}

bool korobeinikov_a_test_task_seq_lab_03::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output

  bool flag = true;
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    if (taskData->inputs[2][i] < 0) {
      flag = false;
    }
  }
  return taskData->inputs.size() == 5 && taskData->inputs_count.size() == 5 && taskData->outputs.size() == 1 &&
         taskData->outputs_count.size() == 1 && flag && *(reinterpret_cast<int *>(taskData->inputs[4])) >= 0 &&
         (*reinterpret_cast<int *>(taskData->inputs[4])) < (*reinterpret_cast<int *>(taskData->inputs[3])) &&
         (int)taskData->outputs_count[0] == *reinterpret_cast<int *>(taskData->inputs[3]) &&
         (int)taskData->inputs_count[2] == *reinterpret_cast<int *>(taskData->inputs[3]) + 1;
}

bool korobeinikov_a_test_task_seq_lab_03::TestTaskSequential::run() {
  internal_order_test();

  std::vector<bool> visited(size, false);
  std::vector<int> D(size, std::numeric_limits<int>::max());
  D[sv] = 0;
  for (int i = 0; i < size; i++) {
    int min = std::numeric_limits<int>::max();
    int index = -1;
    for (int j = 0; j < size; j++) {
      if (!visited[j] && D[j] < min) {
        min = D[j];
        index = j;
      }
    }
    if (index == -1) break;
    visited[index] = true;
    for (int j = RowIndex[index]; j < RowIndex[index + 1]; j++) {
      int v = col[j];
      int weight = values[j];

      if (!visited[v] && (D[index] + weight < D[v])) {
        D[v] = D[index] + weight;
      }
    }
  }
  res = D;
  return true;
}

bool korobeinikov_a_test_task_seq_lab_03::TestTaskSequential::post_processing() {
  internal_order_test();
  std::copy(res.begin(), res.end(), reinterpret_cast<int *>(taskData->outputs[0]));
  return true;
}
