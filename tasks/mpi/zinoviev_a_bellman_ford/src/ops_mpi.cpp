// Copyright 2023 Nesterov Alexander
#include "mpi/zinoviev_a_bellman_ford/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel::pre_processing() {
  internal_order_test();

  // Initialize row pointers, column indices, and values
  row_pointers_ = std::vector<int>(taskData->inputs_count[0]);
  col_indices_ = std::vector<int>(taskData->inputs_count[1]);
  values_ = std::vector<int>(taskData->inputs_count[2]);

  auto* row_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  auto* col_idx = reinterpret_cast<int*>(taskData->inputs[1]);
  auto* val = reinterpret_cast<int*>(taskData->inputs[2]);

  for (size_t i = 0; i < row_pointers_.size(); ++i) {
    row_pointers_[i] = row_ptr[i];
  }
  for (size_t i = 0; i < col_indices_.size(); ++i) {
    col_indices_[i] = col_idx[i];
  }
  for (size_t i = 0; i < values_.size(); ++i) {
    values_[i] = val[i];
  }

  // Initialize distances
  distances_ = std::vector<int>(taskData->outputs_count[0], INT_MAX);
  distances_[0] = 0;  // Source node
  return true;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel::validation() {
  internal_order_test();
  return taskData->outputs_count[0] > 0;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel::run() {
  internal_order_test();

  // Bellman-Ford algorithm
  for (size_t i = 0; i < distances_.size() - 1; ++i) {
    for (size_t u = 0; u < row_pointers_.size() - 1; ++u) {
      for (size_t j = row_pointers_[u]; j < row_pointers_[u + 1]; ++j) {
        int v = col_indices_[j];
        int weight = values_[j];
        if (distances_[u] != INT_MAX && distances_[u] + weight < distances_[v]) {
          distances_[v] = distances_[u] + weight;
        }
      }
    }
    // Reduce distances across all processes
    boost::mpi::reduce(
        world, distances_.data(), distances_.size(), distances_.data(),
        [](const int* in, const int* in_end, int* out) {
          std::transform(in, in_end, out, out, [](int a, int b) { return std::min(a, b); });
        },
        0);
  }
  return true;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < distances_.size(); ++i) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = distances_[i];
    }
  }
  return true;
}