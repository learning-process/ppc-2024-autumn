
#include "seq/strakhov_a_str_char_freq/include/ops_seq.hpp"

#include <algorithm>
#include <string>
#include <thread>
using namespace std::chrono_literals;

bool strakhov_a_str_char_freq_seq::TaskStringCharactersFrequencySequential::pre_processing() {
  internal_order_test();
  input_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  target_ = *reinterpret_cast<char*>(taskData->inputs[1]);
  res = 0;
  return true;
}

bool strakhov_a_str_char_freq_seq::TaskStringCharactersFrequencySequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] == 1) && (taskData->inputs_count[1] == 1) && (taskData->outputs_count[0] == 1);
}

bool strakhov_a_str_char_freq_seq::TaskStringCharactersFrequencySequential::run() {
  internal_order_test();
  res = std::count(input_.begin(), input_.end(), target_) / input_.length();
  return true;
}

bool strakhov_a_str_char_freq_seq::TaskStringCharactersFrequencySequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = res;
  return true;
}