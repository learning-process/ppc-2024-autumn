// Copyright 2024 Nesterov Alexander
#include "seq/lavrentyev_a_alternation_count/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool lavrentyev_a_alternation_count_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  auto input_size = taskData->inputs_count[0];
  input_ = std::vector<int>(input_ptr, input_ptr + input_size);
  res = 0;
  return true;
}

bool lavrentyev_a_alternation_count_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool lavrentyev_a_alternation_count_seq::TestTaskSequential::run() {
  internal_order_test();
  for (unsigned long i = 0; i < input_.size() - 1; i++) {
    if (((input_[i] * input_[i + 1]) <= 0) && (input_[i] != 0 || input_[i + 1] != 0)) {
      res++;
    }
  }
  return true;
}

bool lavrentyev_a_alternation_count_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}