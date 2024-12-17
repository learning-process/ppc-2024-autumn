// Copyright 2023 Nesterov Alexander
#include "mpi/zinoviev_a_bellman_ford/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskSequential::pre_processing() {
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

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] > 0;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskSequential::run() {
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

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < distances_.size(); ++i) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = distances_[i];
  }
  return true;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init graph
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
  // Init distances
  distances_ = std::vector<int>(taskData->outputs_count[0], INT_MAX);
  distances_[0] = 0;
  return true;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] > 0;
  }
  return true;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel::run() {
  internal_order_test();
  // Bellman-Ford algorithm
  for (size_t i = 0; i < distances_.size() - 1; ++i) {
    for (size_t j = 0; j < local_graph_.size(); j += 3) {
      int u = local_graph_[j];
      int v = local_graph_[j + 1];
      int weight = local_graph_[j + 2];
      if (distances_[u] != INT_MAX && distances_[u] + weight < distances_[v]) {
        distances_[v] = distances_[u] + weight;
      }
    }
    // Reduce distances
    std::vector<int> local_distances = distances_;
    boost::mpi::reduce(
        world, local_distances.data(), local_distances.size(), distances_.data(),
        [](int a, int b) { return std::min(a, b); }, 0);
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