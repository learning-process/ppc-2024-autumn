#include "seq/kovalchuk_a_horizontal_tape_scheme/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

bool kovalchuk_a_horizontal_tape_scheme_seq::TestSequentialTask::pre_processing() {
  internal_order_test();
  // Init matrix and vector
  if (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0) {
    matrix_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], matrix_[i].begin());
    }
    vector_ = std::vector<int>(taskData->inputs_count[1]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[taskData->inputs_count[0]]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], vector_.begin());
  } else {
    matrix_ = std::vector<std::vector<int>>();
    vector_ = std::vector<int>();
  }
  // Init result vector
  result_ = std::vector<int>(taskData->inputs_count[0], 0);
  return true;
}

bool kovalchuk_a_horizontal_tape_scheme_seq::TestSequentialTask::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool kovalchuk_a_horizontal_tape_scheme_seq::TestSequentialTask::run() {
  internal_order_test();
  if (!matrix_.empty() && !vector_.empty()) {
    for (unsigned int i = 0; i < matrix_.size(); i++) {
      result_[i] = std::inner_product(matrix_[i].begin(), matrix_[i].end(), vector_.begin(), 0);
    }
  }
  return true;
}

bool kovalchuk_a_horizontal_tape_scheme_seq::TestSequentialTask::post_processing() {
  internal_order_test();
  std::copy(result_.begin(), result_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}
