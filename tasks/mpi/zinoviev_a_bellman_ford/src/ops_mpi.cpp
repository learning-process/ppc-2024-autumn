// Copyright 2023 Nesterov Alexander
#include "mpi/zinoviev_a_bellman_ford/include/ops_mpi.hpp"

#include <algorithm>
#include <random>
#include <vector>

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    graph_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      graph_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, graph_.data() + proc * delta, delta);
    }
  }
  local_graph_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    local_graph_ = std::vector<int>(graph_.begin(), graph_.begin() + delta);
  } else {
    world.recv(0, 0, local_graph_.data(), delta);
  }
  dist_ = std::vector<int>(taskData->outputs_count[0], 0);
  return true;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] > 0;
  }
  return true;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel::run() {
  internal_order_test();
  // Implement Bellman-Ford algorithm here
  // This is a placeholder for the actual implementation
  return true;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    for (unsigned i = 0; i < dist_.size(); i++) {
      tmp_ptr[i] = dist_[i];
    }
  }
  return true;
}