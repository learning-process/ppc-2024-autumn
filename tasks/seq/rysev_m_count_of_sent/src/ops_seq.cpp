// Copyright 2024 Nesterov Alexander
#include "seq/rysev_m_count_of_sent/include/ops_seq.hpp"

bool rysev_m_count_of_sent_seq::SentCountSequential::pre_processing() {
  internal_order_test();
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  count = 0;
  return true;
}

bool rysev_m_count_of_sent_seq::SentCountSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool rysev_m_count_of_sent_seq::SentCountSequential::run() {
  internal_order_test();
  char last_symbol = ' ';
  for (char symbol : input_) {
    if ((symbol == '.' || symbol == '!' || symbol == '?') && last_symbol != '.' && last_symbol != '!' && last_symbol != '?') count += 1;
    last_symbol = symbol;
  }
  if (input_.back() != '.' && input_.back() != '!' && input_.back() != '?' && !input_.empty()) count += 1;
  return true;
}

bool rysev_m_count_of_sent_seq::SentCountSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = count;
  return true;
}
