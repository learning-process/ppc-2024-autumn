#include "seq/shvedova_v_char_freq/include/ops_seq.hpp"

#include <algorithm>
#include <string>
#include <thread>

using namespace std::chrono_literals;

bool shvedova_v_char_frequency_seq::CharFrequencyTaskSequential::pre_processing() {
  internal_order_test();
  input_str_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  target_char_ = *reinterpret_cast<char*>(taskData->inputs[1]);
  frequency_ = 0;
  return true;
}

bool shvedova_v_char_frequency_seq::CharFrequencyTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 1 && taskData->inputs_count[1] == 1 && taskData->outputs_count[0] == 1;
}

bool shvedova_v_char_frequency_seq::CharFrequencyTaskSequential::run() {
  internal_order_test();
  frequency_ = std::count(input_str_.begin(), input_str_.end(), target_char_);
  return true;
}

bool shvedova_v_char_frequency_seq::CharFrequencyTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = frequency_;
  return true;
}