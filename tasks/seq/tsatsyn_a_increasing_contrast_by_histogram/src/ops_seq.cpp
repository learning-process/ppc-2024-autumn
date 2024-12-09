// Copyright 2024 Nesterov Alexander
#include "seq/tsatsyn_a_increasing_contrast_by_histogram/include/ops_seq.hpp"

#include <thread>


bool tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = reinterpret_cast<int*>(taskData->inputs[0])[0];
  res = 0;
  return true;
}

bool tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < input_; i++) {
    res++;
  }
  return true;
}

bool tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
