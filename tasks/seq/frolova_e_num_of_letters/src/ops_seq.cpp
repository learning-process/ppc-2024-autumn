// Copyright 2024 Nesterov Alexander
#include "seq/frolova_e_num_of_letters/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

std::string frolova_e_num_of_letters_seq::GenStr(int n) {
  std::string str = "test";

  std::string result;
  result.resize(n);

  int i = 0;
  size_t j = 0;

  while (i < n) {
    result[i] = str[j];
    j++;
    i++;
    if (j >= str.size()) {
      j = 0;
    }
  }

  return result;
}

int frolova_e_num_of_letters_seq::Count(std::string& str) {
  int count = 0;
  for (char c : str) {
    if (static_cast<bool>(isalpha(c))) {
      count++;
    }
  }
  return count;
}

bool frolova_e_num_of_letters_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  res = 0;
  return true;
}

bool frolova_e_num_of_letters_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool frolova_e_num_of_letters_seq::TestTaskSequential::run() {
  internal_order_test();
  res = Count(input_);
  return true;
}

bool frolova_e_num_of_letters_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}