// Copyright 2024 Nesterov Alexander
#include "seq/zinoviev_a_bellman_ford/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
#include <vector>

bool zinoviev_a_bellman_ford_seq::BellmanFordTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  res = 0;
  return true;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordTaskSequential::run() {
  internal_order_test();
  const int num_vertices = input_[0];
  const int num_edges = input_[1];
  const int* row_ptr = input_.data() + 2;
  const int* col_ind = input_.data() + 2 + num_vertices + 1;
  const int* values = input_.data() + 2 + num_vertices + 1 + num_edges;

  std::vector<int> dist(num_vertices, std::numeric_limits<int>::max());
  dist[0] = 0;

  for (int i = 0; i < num_vertices - 1; i++) {
    for (int u = 0; u < num_vertices; u++) {
      for (int j = row_ptr[u]; j < row_ptr[u + 1]; j++) {
        int v = col_ind[j];
        int weight = values[j];
        if (dist[u] != std::numeric_limits<int>::max() && dist[u] + weight < dist[v]) {
          dist[v] = dist[u] + weight;
        }
      }
    }
  }

  for (int u = 0; u < num_vertices; u++) {
    for (int j = row_ptr[u]; j < row_ptr[u + 1]; j++) {
      int v = col_ind[j];
      int weight = values[j];
      if (dist[u] != std::numeric_limits<int>::max() && dist[u] + weight < dist[v]) {
        res = -1;
        return true;
      }
    }
  }

  res = dist[num_vertices - 1];
  return true;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}