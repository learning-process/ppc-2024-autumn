// Copyright 2024 Nesterov Alexander
#include "seq/opolin_d_max_of_matrix_elements/include/ops_seq.hpp"

#include <climits>
#include <random>

using namespace std::chrono_literals;

bool opolin_d_max_of_matrix_elements_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init matrix
  input_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  // Init value for output
  res = std::numeric_limits<int32_t>::min();
  return true;
}

bool opolin_d_max_of_matrix_elements_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check non empty input
  if (static_cast<int>(taskData->inputs_count[0]) <= 0 || static_cast<int>(taskData->inputs_count[1]) <= 0) {
    return false;
  }
  // Check output
  if (taskData->outputs_count[0] != 1) {
    return false;
  }
  return true;
}

bool opolin_d_max_of_matrix_elements_seq::TestTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); i++) {
    for (size_t j = 0; j < input_[i].size(); j++) {
      if (input_[i][j] > res) {
        res = input_[i][j];
      }
    }
  }
  return true;
}

bool opolin_d_max_of_matrix_elements_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}