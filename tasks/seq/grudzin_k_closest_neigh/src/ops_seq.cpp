// Copyright 2024 Nesterov Alexander
#include "seq/grudzin_k_closest_neigh/include/ops_seq.hpp"

#include <thread>
#include <climits>

using namespace std::chrono_literals;

bool grudzin_k_closest_neigh_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::vector<int>(taskData->inputs_count[0]);
  int* mas = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = mas[i];
  }
  res = INT_MAX;
  return true;
}

bool grudzin_k_closest_neigh_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 1 && taskData->outputs_count[0] == 1;
}

bool grudzin_k_closest_neigh_seq::TestTaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < input_.size()-1; i++) {
    res = std::min(res, std::abs(input_[i] - input_[i + 1]));
  }
  //std::cout << res << std::endl;
  return true;
}

bool grudzin_k_closest_neigh_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
