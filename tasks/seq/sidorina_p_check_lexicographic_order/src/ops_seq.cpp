// Copyright 2024 Nesterov Alexander
#include "seq/sidorina_p_check_lexicographic_order/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool sidorina_p_check_lexicographic_order_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_.resize(taskData->inputs_count[0], std::vector<char>(taskData->inputs_count[1]));
  for (size_t i = 0; i < taskData->inputs_count[0]; ++i) {
    const char* tmp_ptr = reinterpret_cast<const char*>(taskData->inputs[i]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], input_[i].begin());
  }
  res = 0;
  return true;
}
bool sidorina_p_check_lexicographic_order_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 2 && taskData->outputs_count[0] == 1;
}

bool sidorina_p_check_lexicographic_order_seq::TestTaskSequential::run() {
  internal_order_test();

  for (size_t i = 0; i < std::min(input_[0].size(), input_[1].size()); ++i) {
    if (input_[0][i] > input_[1][i]) {
      res = 1;
      break;
    }
    if (input_[0][i] < input_[1][i]) break;                        
  }
  return true;
}
bool sidorina_p_check_lexicographic_order_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
