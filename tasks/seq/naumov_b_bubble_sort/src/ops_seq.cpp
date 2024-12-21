// Copyright 2024 Nesterov Alexander
#include "seq/naumov_b_bubble_sort/include/ops_seq.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

namespace naumov_b_bubble_sort_seq {

bool TestTaskSequential::pre_processing() {
  internal_order_test();

  auto* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int input_size = taskData->inputs_count[0];
  input_.resize(input_size);
  std::copy(input_data, input_data + input_size, input_.begin());

  return true;
}

bool TestTaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }
  if (taskData->inputs_count[0] <= 0) {
    return false;
  }
  if (taskData->outputs_count.empty() || taskData->outputs_count[0] != taskData->inputs_count[0]) {
    return false;
  }

  return true;
}

bool TestTaskSequential::run() {
  internal_order_test();

  for (size_t i = 0; i < input_.size(); ++i) {
    for (size_t j = 0; j < input_.size() - i - 1; ++j) {
      if (input_[j] > input_[j + 1]) {
        std::swap(input_[j], input_[j + 1]);
      }
    }
  }

  return true;
}

bool TestTaskSequential::post_processing() {
  internal_order_test();

  if (taskData->outputs.empty() || taskData->outputs_count[0] != input_.size()) {
    return false;
  }

  auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(input_.begin(), input_.end(), output_data);

  return true;
}

}  // namespace naumov_b_bubble_sort_seq
