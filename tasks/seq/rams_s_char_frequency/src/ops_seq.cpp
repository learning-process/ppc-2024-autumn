// Copyright 2024 Nesterov Alexander
#include "seq/rams_s_char_frequency/include/ops_seq.hpp"

#include <algorithm>
#include <string>

using namespace std::chrono_literals;

bool rams_s_char_frequency_seq::CharFrequencyTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  target_ = *reinterpret_cast<char*>(taskData->inputs[1]);
  res = 0;
  return true;
}

bool rams_s_char_frequency_seq::CharFrequencyTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] == 1 && taskData->outputs_count[0] == 1;
}

bool rams_s_char_frequency_seq::CharFrequencyTaskSequential::run() {
  internal_order_test();
  res = std::count(input_.begin(), input_.end(), target_);
  return true;
}

bool rams_s_char_frequency_seq::CharFrequencyTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
