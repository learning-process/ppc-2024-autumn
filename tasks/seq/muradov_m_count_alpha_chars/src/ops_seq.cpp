#include "seq/muradov_m_count_alpha_chars/include/ops_seq.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <thread>

using namespace std::chrono_literals;

bool muradov_m_count_alpha_chars_seq::AlphaCharCountTaskSequential::pre_processing() {
  internal_order_test();
  input_str_ = *reinterpret_cast<std::string*>(taskData->inputs[0]);
  alpha_count_ = 0;
  return true;
}

bool muradov_m_count_alpha_chars_seq::AlphaCharCountTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool muradov_m_count_alpha_chars_seq::AlphaCharCountTaskSequential::run() {
  internal_order_test();
  alpha_count_ = std::count_if(input_str_.begin(), input_str_.end(),
                               [](char c) { return std::isalpha(static_cast<unsigned char>(c)); });
  std::this_thread::sleep_for(20ms);
  return true;
}

bool muradov_m_count_alpha_chars_seq::AlphaCharCountTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<int*>(taskData->outputs[0]) = alpha_count_;
  return true;
}
