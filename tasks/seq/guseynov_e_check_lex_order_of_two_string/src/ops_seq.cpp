#include "seq/guseynov_e_check_lex_order_of_two_string/include/ops_seq.hpp"

bool guseynov_e_check_lex_order_of_two_string_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<std::vector<char>>(taskData->inputs_count[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    auto *tmp_ptr = reinterpret_cast<char *>(taskData->inputs[i]);
    input_[i] = std::vector<char>(taskData->inputs_count[i + 1]);
    for (unsigned j = 0; j < taskData->inputs_count[i + 1]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }

  res_ = 0;
  return true;
}

bool guseynov_e_check_lex_order_of_two_string_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 2 && taskData->outputs_count[0] == 1;
}

bool guseynov_e_check_lex_order_of_two_string_seq::TestTaskSequential::run() {
  internal_order_test();
  size_t min_string_len = std::min(input_[0].size(), input_[1].size());
  for (size_t i = 0; i < min_string_len; i++) {
    if (input_[0][i] < input_[1][i]) {
      res_ = 1;
      break;
    }
    if (input_[0][i] > input_[1][i]) {
      res_ = 2;
      break;
    }
  }
  if (res_ == 0 && input_[0].size() != input_[1].size()) {
    if (input_[0].size() > input_[1].size()) {
      res_ = 2;
    } else {
      res_ = 1;
    }
  }
  return true;
}

bool guseynov_e_check_lex_order_of_two_string_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = res_;
  return true;
}