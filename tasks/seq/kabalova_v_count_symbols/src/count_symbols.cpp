// Copyright 2024 Nesterov Alexander
#include "seq/kabalova_v_count_symbols/include/count_symbols.hpp"

#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int kabalova_v_count_symbols_seq::getRandomNumber(int left, int right) {
  return ((rand() % (right - left + 1)) + left);
}

std::string kabalova_v_count_symbols_seq::getRandomString() {
  srand(time(NULL));
  std::string str;
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz1234567890";
  int strSize = getRandomNumber(1000, 10000);
  for (int i = 0; i < strSize; i++) {
    str += alphabet[getRandomNumber(0, alphabet.size() - 1)];
  }
  return str;
}

int kabalova_v_count_symbols_seq::countSymbols(std::string& str) {
  int result = 0;
  for (size_t i = 0; i < str.size(); i++) {
    if (isalpha(str[i])) {
      result++;
    }
  }
  return result;
};

bool kabalova_v_count_symbols_seq::Task1Seq::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  result = 0;
  return true;
}

bool kabalova_v_count_symbols_seq::Task1Seq::validation() {
  internal_order_test();
  // На выход подается 1 строка, на выходе только 1 число - число буквенных символов в строке.
  bool flag1 = (taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1);
  // Нам пришел массив char'ов?
  bool flag2 = false;
  if (typeid(*taskData->inputs[0]).name() == typeid(uint8_t).name()) {
    flag2 = true;
  }
  return (flag1 && flag2);
}

bool kabalova_v_count_symbols_seq::Task1Seq::run() {
  internal_order_test();
  result = countSymbols(input_);
  return true;
}

bool kabalova_v_count_symbols_seq::Task1Seq::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
  return true;
}
