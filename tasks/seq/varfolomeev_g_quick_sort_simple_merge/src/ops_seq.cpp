// Copyright 2024 Nesterov Alexander
#include "seq/varfolomeev_g_quick_sort_simple_merge/include/ops_seq.hpp"

bool varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  res = input_;
  return true;
}

bool varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential::validation() {
  internal_order_test();
  return (taskData->outputs_count.size() == 1 && taskData->inputs_count.size() == 1 && taskData->inputs_count[0] > 0 &&
          taskData->inputs_count[0] == taskData->outputs_count[0]);
}

bool varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential::run() {
  internal_order_test();
  quickSort(res);
  return true;
}

bool varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res.begin(), res.end(), output_ptr);
  return true;
}
