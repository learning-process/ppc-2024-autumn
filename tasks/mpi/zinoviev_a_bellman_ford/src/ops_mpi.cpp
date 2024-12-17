// Copyright 2023 Nesterov Alexander
#include "mpi/zinoviev_a_bellman_ford/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

void zinoviev_a_bellman_ford_mpi::BellmanFordMPISeq::toCRS(const int* input_matrix) {
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

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPISeq::pre_processing() {
  internal_order_test();
  auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
  V = taskData->inputs_count[0];
  E = taskData->inputs_count[1];

  toCRS(input_matrix);

  shortest_paths.resize(V, INF);
  shortest_paths[0] = 0;
  return true;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPISeq::validation() {
  internal_order_test();
  return taskData->inputs_count[0] < taskData->inputs_count[1] &&
         taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPISeq::Iteration(std::vector<int>& paths) {
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

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPISeq::check_negative_cycle() {
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

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPISeq::run() {
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

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPISeq::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < V; ++i) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = shortest_paths[i];
  }
  return true;
}

// MPI Implementation

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPIMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
    V = taskData->inputs_count[0];
    E = taskData->inputs_count[1];

    toCRS(input_matrix);
  }
  boost::mpi::broadcast(world, V, 0);
  boost::mpi::broadcast(world, values, 0);
  boost::mpi::broadcast(world, columns, 0);
  boost::mpi::broadcast(world, row_ptr, 0);

  shortest_paths.resize(V, INF);
  shortest_paths[0] = 0;
  return true;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPIMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] < taskData->inputs_count[1] &&
           taskData->inputs_count[0] == taskData->outputs_count[0];
  }
  return true;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPIMPI::Iteration(std::vector<int>& paths) {
  bool changed = false;
  std::vector<int> start_paths = paths;
  int rank = world.rank();
  int size = world.size();

  int local_size = V / size;
  int remainder = V % size;
  int start = rank * local_size + std::min(rank, remainder);
  int end = start + local_size + (rank < remainder ? 1 : 0);

  for (int i = start; i < end; i++) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      int v = columns[j];
      int weight = values[j];
      if (start_paths[i] != INF && start_paths[i] + weight < start_paths[v]) {
        start_paths[v] = start_paths[i] + weight;
      }
    }
  }

  std::vector<int> reduced_paths(V, INF);
  boost::mpi::reduce(
      world, start_paths.data(), V, reduced_paths.data(),
      [](int a, int b) {
        if (a == zinoviev_a_bellman_ford_mpi::BellmanFordMPIMPI::INF) return b;
        if (b == zinoviev_a_bellman_ford_mpi::BellmanFordMPIMPI::INF) return a;
        return std::min(a, b);
      },
      0);

  if (world.rank() == 0) {
    for (size_t i = 0; i < V; i++) {
      if (paths[i] != reduced_paths[i]) {
        changed = true;
        break;
      }
    }
    paths = reduced_paths;
  }

  boost::mpi::broadcast(world, paths.data(), paths.size(), 0);
  boost::mpi::broadcast(world, changed, 0);

  return changed;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPIMPI::check_negative_cycle() {
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

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPIMPI::run() {
  internal_order_test();

  bool changed = false;
  for (size_t i = 0; i < V - 1; ++i) {
    changed = Iteration(shortest_paths);
    if (!changed) break;
  }

  if (world.rank() == 0) {
    for (size_t i = 0; i < V; ++i) {
      if (shortest_paths[i] == INF) shortest_paths[i] = 0;
    }
  }

  if (world.rank() == 0) {
    return !check_negative_cycle();
  }

  return true;
}

bool zinoviev_a_bellman_ford_mpi::BellmanFordMPIMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < V; ++i) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = shortest_paths[i];
    }
  }
  return true;
}