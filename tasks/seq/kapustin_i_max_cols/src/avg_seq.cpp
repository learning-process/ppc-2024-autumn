// Copyright 2024 Nesterov Alexander
#include "seq/kapustin_i_max_cols/include/avg_seq.hpp"
#include <algorithm>
#include <functional>

using namespace std::chrono_literals;

bool kapustin_i_max_column_task_seq::MaxColumnTaskSequential::pre_processing() {
  internal_order_test();
  column_count = *reinterpret_cast<int*>(taskData->inputs[1]);
  int total_elements = taskData->inputs_count[0];
  row_count = total_elements / column_count;
  input_.resize(total_elements);
  auto* matrix_data = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < total_elements; ++i) {
    input_[i] = matrix_data[i];
  }
  res.resize(column_count, std::numeric_limits<int>::min());
  return true;
}

bool kapustin_i_max_column_task_seq::MaxColumnTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count.size() < 2 || taskData->inputs_count[0] == 0 || taskData->inputs_count[1] == 0)
    return false;
  row_count = *reinterpret_cast<int*>(taskData->inputs[1]);
  if (row_count <= 0) return false;
  if (taskData->inputs_count[0] % row_count != 0) return false;
  return true;
}

bool kapustin_i_max_column_task_seq::MaxColumnTaskSequential::run() {
  internal_order_test();
  for (int j = 0; j < column_count; ++j) {
    int max_value = std::numeric_limits<int>::min();
    for (int i = 0; i < row_count; ++i) {
      int current_value = input_[i * column_count + j];
      if (current_value > max_value) {
        max_value = current_value;
      }
    }
    res[j] = max_value;
  }
  return true;
}

bool kapustin_i_max_column_task_seq::MaxColumnTaskSequential::post_processing() {
  internal_order_test();
  for (int j = 0; j < column_count; ++j) {
    reinterpret_cast<int*>(taskData->outputs[0])[j] = res[j];
  }
  return true;
}
