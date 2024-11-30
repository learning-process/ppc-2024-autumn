// Copyright 2024 Nesterov Alexander
#include "seq/kabalova_v_count_symbols/include/count_symbols.hpp"

#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int kabalova_v_count_symbols_seq::countSymbols(std::string& str) {
  int result = 0;
  for (size_t i = 0; i < str.size(); i++) {
    if (isalpha(str[i]) != 0) {
      result++;
    }
  }
  return result;
}

bool kabalova_v_count_symbols_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  result = 0;
  return true;
}

bool kabalova_v_count_symbols_seq::TestTaskSequential::validation() {
  internal_order_test();
  // На выход подается 1 строка, на выходе только 1 число - число буквенных символов в строке.
  return (taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1);
}

bool kabalova_v_count_symbols_seq::TestTaskSequential::run() {
  internal_order_test();
  result = countSymbols(input_);
  return true;
}

bool kabalova_v_count_symbols_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  return true;
}
