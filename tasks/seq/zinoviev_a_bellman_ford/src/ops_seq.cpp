// Copyright 2024 Nesterov Alexander
#include "seq/zinoviev_a_bellman_ford/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

void zinoviev_a_bellman_ford_seq::BellmanFordSeq::toCRS(const int* input_matrix) {
  row_ptr.push_back(0);
  for (size_t i = 0; i < V; ++i) {
    for (size_t j = 0; j < V; ++j) {
      if (input_matrix[i * V + j] != 0) {
        values.push_back(input_matrix[i * V + j]);
        columns.push_back(j);
      }
    }
    row_ptr.push_back(values.size());
  }
}

bool zinoviev_a_bellman_ford_seq::BellmanFordSeq::pre_processing() {
  internal_order_test();
  auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
  V = taskData->inputs_count[0];
  E = taskData->inputs_count[1];

  toCRS(input_matrix);

  shortest_paths.resize(V, INF);
  shortest_paths[0] = 0;
  return true;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordSeq::validation() {
  internal_order_test();
  return taskData->inputs_count[0] < taskData->inputs_count[1] &&
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool zinoviev_a_bellman_ford_seq::BellmanFordSeq::Iteration(std::vector<int>& paths) {
  bool changed = false;
  for (size_t i = 0; i < V; ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      int v = columns[j];
      int weight = values[j];
      if (paths[i] != INF && paths[i] + weight < paths[v]) {
        paths[v] = paths[i] + weight;
        changed = true;
      }
    }
  }
  return changed;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordSeq::check_negative_cycle() {
  for (size_t i = 0; i < V; ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      int v = columns[j];
      int weight = values[j];
      if (shortest_paths[i] != INF && shortest_paths[i] + weight < shortest_paths[v]) {
        return true;
      }
    }
  }
  return false;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordSeq::run() {
  internal_order_test();

  bool changed = false;
  for (size_t i = 0; i < V - 1; ++i) {
    changed = Iteration(shortest_paths);
    if (!changed) break;
  }

  if (check_negative_cycle()) {
    return false;
  }

  for (size_t i = 0; i < V; ++i) {
    if (shortest_paths[i] == INF) shortest_paths[i] = 0;
  }
  return true;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordSeq::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < V; ++i) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = shortest_paths[i];
  }
  return true;
}