#include "seq/sarafanov_m_num_of_mismatch_characters_of_two_strings/include/ops_seq.hpp"

bool sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->inputs_count[1] && taskData->outputs_count[0] == 1;
}

bool sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask::pre_processing() {
  internal_order_test();
  input_a_.assign(reinterpret_cast<char *>(taskData->inputs[0]), taskData->inputs_count[0]);
  input_b_.assign(reinterpret_cast<char *>(taskData->inputs[1]), taskData->inputs_count[1]);
  result_ = 0;
  return true;
}

bool sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask::run() {
  internal_order_test();
  for (size_t i = 0; i < input_a_.size(); ++i) {
    if (input_a_[i] != input_b_[i]) {
      result_++;
    }
  }
  return true;
}

bool sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask::post_processing() {
  internal_order_test();
  *reinterpret_cast<int *>(taskData->outputs[0]) = result_;
  return true;
}
