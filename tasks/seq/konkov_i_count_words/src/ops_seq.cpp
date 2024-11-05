// Copyright 2023 Konkov Ivan
#include "seq/konkov_i_count_words/include/ops_seq.hpp"

#include <sstream>

bool konkov_i_count_words_seq::CountWordsTaskSequential::pre_processing() {
  internal_order_test();
  input_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  word_count_ = 0;
  return true;
}

bool konkov_i_count_words_seq::CountWordsTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1 && taskData->inputs[0] != nullptr &&
         taskData->outputs[0] != nullptr;
}

bool konkov_i_count_words_seq::CountWordsTaskSequential::run() {
  internal_order_test();
  std::istringstream stream(input_);
  std::string word;
  while (stream >> word) {
    word_count_++;
  }
  return true;
}

bool konkov_i_count_words_seq::CountWordsTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = word_count_;
  return true;
}