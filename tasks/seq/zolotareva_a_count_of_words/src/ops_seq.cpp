// Copyright 2024 Nesterov Alexander
#include "seq/zolotareva_a_count_of_words/include/ops_seq.hpp"

#include <string>

bool zolotareva_a_count_of_words_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool zolotareva_a_count_of_words_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_.assign(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  res = 0;
  return true;
}

bool zolotareva_a_count_of_words_seq::TestTaskSequential::run() {
  internal_order_test();

  bool in_word = false;
  for (char c : input_) {
    if (c == ' ')
      in_word = false;
    else if (!in_word) {
      ++res;
      in_word = true;
    }
  }
  return true;
}

bool zolotareva_a_count_of_words_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
