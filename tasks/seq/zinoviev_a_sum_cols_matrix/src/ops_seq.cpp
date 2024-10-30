// Copyright 2024 Nesterov Alexander
#include "seq/zinoviev_a_sum_cols_matrix/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = reinterpret_cast<int*>(taskData->inputs[0])[0];
  res = 0;
  return true;
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < input_; i++) {
    res++;
  }
  return true;
}

bool zinoviev_a_sum_cols_matrix::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
