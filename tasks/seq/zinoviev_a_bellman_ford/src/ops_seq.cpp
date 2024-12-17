// Copyright 2024 Nesterov Alexander
#include "seq/zinoviev_a_bellman_ford/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool zinoviev_a_bellman_ford_seq::BellmanFordSeqTaskSequential::pre_processing() {
  internal_order_test();
  // Init graph
  graph_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    graph_[i] = tmp_ptr[i];
  }
  // Init distances
  distances_ = std::vector<int>(taskData->outputs_count[0], INT_MAX);
  distances_[0] = 0;
  return true;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordSeqTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] > 0;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordSeqTaskSequential::run() {
  internal_order_test();
  // Bellman-Ford algorithm
  for (size_t i = 0; i < distances_.size() - 1; ++i) {
    for (size_t j = 0; j < graph_.size(); j += 3) {
      int u = graph_[j];
      int v = graph_[j + 1];
      int weight = graph_[j + 2];
      if (distances_[u] != INT_MAX && distances_[u] + weight < distances_[v]) {
        distances_[v] = distances_[u] + weight;
      }
    }
  }
  return true;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordSeqTaskSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < distances_.size(); ++i) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = distances_[i];
  }
  return true;
}