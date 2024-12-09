#include "../include/ops_seq.hpp"

#include <cstddef>

bool vedernikova_k_word_num_in_str_seq::TestTaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == 1;
}

bool vedernikova_k_word_num_in_str_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  input_.assign(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);

  res_ = 0;

  return true;
}

bool vedernikova_k_word_num_in_str_seq::TestTaskSequential::run() {
  internal_order_test();

  bool is_space = input_[0] == ' ';
  for (const char c : input_) {
    if (c == ' ') {
      if (!is_space) {
        res_++;
      }
      is_space = true;
      continue;
    }
    is_space = false;
  }
  res_ += (is_space || input_.empty()) ? 0 : 1;

  return true;
}

bool vedernikova_k_word_num_in_str_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  *reinterpret_cast<size_t*>(taskData->outputs[0]) = res_;

  return true;
}
