// Copyright 2024 Nesterov Alexander
#include "seq/varfolomeev_g_quick_sort_simple_merge/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = reinterpret_cast<int*>(taskData->inputs[0])[0];
  res = 0;
  return true;
}

bool varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < input_; i++) {
    res++;
  }
  std::this_thread::sleep_for(20ms);
  return true;
}

bool varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
