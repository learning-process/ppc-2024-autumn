// Copyright 2024 Nesterov Alexander
#include "seq/zinoviev_a_bellman_ford/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

bool zinoviev_a_bellman_ford_seq::BellmanFordSeqTaskSequential::pre_processing() {
  internal_order_test();
  graph_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    graph_[i] = tmp_ptr[i];
  }
  dist_ = std::vector<int>(taskData->outputs_count[0], 0);
  return true;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordSeqTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] > 0;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordSeqTaskSequential::run() {
  internal_order_test();
  return true;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordSeqTaskSequential::post_processing() {
  internal_order_test();
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  for (unsigned i = 0; i < dist_.size(); i++) {
    tmp_ptr[i] = dist_[i];
  }
  return true;
}