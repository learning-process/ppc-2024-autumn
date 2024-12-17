// Copyright 2024 Nesterov Alexander
#include "seq/zinoviev_a_bellman_ford/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
#include <vector>

bool zinoviev_a_bellman_ford_seq::BellmanFordTaskSequential::pre_processing() {
  internal_order_test();
  // Initialize input data
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Initialize result
  res = 0;
  return true;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordTaskSequential::validation() {
  internal_order_test();
  // Check that the output size is correct
  return taskData->outputs_count[0] == 1;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordTaskSequential::run() {
  internal_order_test();
  // Example implementation of Bellman-Ford algorithm
  // Assuming input_ is a graph in CRS format: [num_vertices, num_edges, row_ptr, col_ind, values]
  const int num_vertices = input_[0];
  const int num_edges = input_[1];
  const int* row_ptr = input_.data() + 2;
  const int* col_ind = input_.data() + 2 + num_vertices + 1;
  const int* values = input_.data() + 2 + num_vertices + 1 + num_edges;

  // Initialize distances
  std::vector<int> dist(num_vertices, std::numeric_limits<int>::max());
  dist[0] = 0;  // Start from the first vertex

  // Relax edges repeatedly
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

  // Check for negative-weight cycles
  for (int u = 0; u < num_vertices; u++) {
    for (int j = row_ptr[u]; j < row_ptr[u + 1]; j++) {
      int v = col_ind[j];
      int weight = values[j];
      if (dist[u] != std::numeric_limits<int>::max() && dist[u] + weight < dist[v]) {
        // Negative cycle detected
        res = -1;
        return true;
      }
    }
  }

  // Store the result (e.g., distance to the last vertex)
  res = dist[num_vertices - 1];
  return true;
}

bool zinoviev_a_bellman_ford_seq::BellmanFordTaskSequential::post_processing() {
  internal_order_test();
  // Write the result to the output
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}