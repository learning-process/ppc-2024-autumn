// Copyright 2024 Nesterov Alexander
#include "seq/kolodkin_g_sentence_count/include/ops_seq.hpp"

#include <thread>

bool kolodkin_g_sentence_count_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  res = 0;
  return true;
}

bool kolodkin_g_sentence_count_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool kolodkin_g_sentence_count_seq::TestTaskSequential::run() {
  internal_order_test();
  for (unsigned long i = 0; i < input_.length(); i++) {
    if ((input_[i] == '.' || input_[i] == '!' || input_[i] == '?') &&
        ((input_[i + 1] != '.' && input_[i + 1] != '!' && input_[i + 1] != '?') || i + 1 == input_.length())) {
      res++;
    }
  }
  return true;
}

bool kolodkin_g_sentence_count_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
