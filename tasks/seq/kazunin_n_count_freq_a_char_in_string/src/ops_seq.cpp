// Copyright 2024 Nesterov Alexander
#include "seq/kazunin_n_count_freq_a_char_in_string/include/ops_seq.hpp"

#include <algorithm>
#include <string>
#include <thread>

namespace kazunin_n_count_freq_a_char_in_string_seq {
bool kazunin_n_count_freq_a_char_in_string_seq::CountFreqCharTaskSequential::pre_processing() {
  internal_order_test();
  input_string_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  target_character_ = *reinterpret_cast<char*>(taskData->inputs[1]);
  frequency_count_ = 0;
  return true;
}

bool kazunin_n_count_freq_a_char_in_string_seq::CountFreqCharTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 1 && taskData->inputs_count[1] == 1 && taskData->outputs_count[0] == 1;
}

bool kazunin_n_count_freq_a_char_in_string_seq::CountFreqCharTaskSequential::run() {
  internal_order_test();
  for (const auto& ch : input_string_) {
    if (ch == target_character_) {
      ++frequency_count_;
    }
  }
  return true;
}

bool kazunin_n_count_freq_a_char_in_string_seq::CountFreqCharTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = frequency_count_;
  return true;
}
}  // namespace kazunin_n_count_freq_a_char_in_string_seq
