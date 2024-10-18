// Copyright 2024 Nesterov Alexander
#include "seq/kolodkin_g_sentence_count/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

int countSentences(const std::string& text) {
  int count = 0;
  for (int i = 0; i < text.length(); i++) {
    if ((text[i] == '.' || text[i] == '!' || text[i] == '?') &&
        ((text[i + 1] != '.' && text[i + 1] != '!' && text[i + 1] != '?') || i + 1 == text.length())) {
      count++;
    }
  }
  return count;
}

bool kolodkin_g_sentence_count_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init string
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  // Init value for output
  res = 0;
  return true;
}

bool kolodkin_g_sentence_count_seq::TestTaskSequential::validation() {
  internal_order_test();
  bool flag1 = (taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1);
  bool flag2 = false;
  if (typeid(*taskData->inputs[0]).name() == typeid(uint8_t).name()) {
    flag2 = true;
  }
  return (flag1 && flag2);
}

bool kolodkin_g_sentence_count_seq::TestTaskSequential::run() {
  internal_order_test();
  res = countSentences(input_);
  return true;
}

bool kolodkin_g_sentence_count_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
