// Copyright 2024 Nesterov Alexander
#include "seq/grudzin_k_nearest_neighbor_elements/include/ops_seq.hpp"

#include <climits>
#include <thread>

using namespace std::chrono_literals;

bool grudzin_k_nearest_neighbor_elements_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  // Init value for output
  res = {INT_MAX, -1};
  return true;
}

bool grudzin_k_nearest_neighbor_elements_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 1 && taskData->outputs_count[0] == 1;
}

bool grudzin_k_nearest_neighbor_elements_seq::TestTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); i++) {
    res = std::min(res, {abs(input_[i] - input_[i + 1]), i});
  }
  return true;
}

bool grudzin_k_nearest_neighbor_elements_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res.second;
  return true;
}
