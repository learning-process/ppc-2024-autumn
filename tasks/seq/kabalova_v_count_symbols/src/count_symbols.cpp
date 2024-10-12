// Copyright 2024 Nesterov Alexander
#include <thread>
#include <random>
#include <vector>

#include "seq/kabalova_v_count_symbols/include/count_symbols.hpp"

using namespace std::chrono_literals;

int kabalova_v_count_symbols_seq::getRandomNumber(int left, int right) {
	return ((rand() % (right - left + 1)) + left);
}

std::string kabalova_v_count_symbols_seq::getRandomString() { 
  srand(time(NULL));
  std::string str = "";
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz1234567890`'!@#$%^&*()-_=+";
  int strSize = getRandomNumber(10, 10000);
  for (int i = 0; i < strSize; i++) {
    str += alphabet[getRandomNumber(0, alphabet.size() - 1)];
  }
  return str;
}

int kabalova_v_count_symbols_seq::countSymbols(std::vector<char> str) { 
  int result = 0;
  for (std::string::size_type i = 0; i < str.size(); i++) {

    if (isalpha(str[i])) {
      result++;
    }
  }
  return result;
};
std::vector<char> kabalova_v_count_symbols_seq::fromStringToChar(std::string& str) {
  std::vector<char> vec;
  for (int i = 0; i < str.size(); i++) {
    vec.emplace_back(str[i]);
  }
  return vec;
}

bool kabalova_v_count_symbols_seq::Task1Seq::pre_processing() {
  internal_order_test();
  // Init value for input and output
  //std::cout << typeid(*reinterpret_cast<uint8_t*>(taskData->inputs[0])).name();
  input_.emplace_back(*reinterpret_cast<uint8_t*>(taskData->inputs[0])); //так работает хот€ бы, но выводит неверные символы
  std::cout << "\nInput[0] =" << input_[0] << "\n";
  result = 0;
  return true;
}

bool kabalova_v_count_symbols_seq::Task1Seq::validation() {
  internal_order_test();
  //Ќа выход подаетс€ 1 строка, на выходе только 1 число - число буквенных символов в строке.
  bool flag1 = (taskData->inputs_count[0] !=0 && taskData->outputs_count[0] == 1);
  //Ќам пришел массив char'ов?
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
