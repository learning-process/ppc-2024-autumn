// Copyright 2024 Nesterov Alexander
#include "seq/stroganov_m_count_symbols_in_string/include/ops_seq.hpp"

#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int stroganov_m_count_symbols_in_string_seq::countSymbols(std::string& str) {
  int result = 0;
  size_t n = str.size();
  for (size_t i = 0; i < n; i++) {
    if (isalpha(str[i]) != 0) {
      result++;
    }
  }
  return result;
}

bool stroganov_m_count_symbols_in_string_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  result = 0;
  return true;
}

bool stroganov_m_count_symbols_in_string_seq::TestTaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1);
}

bool stroganov_m_count_symbols_in_string_seq::TestTaskSequential::run() {
  internal_order_test();
  result = countSymbols(input_);
  return true;
}

bool stroganov_m_count_symbols_in_string_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  return true;
}
