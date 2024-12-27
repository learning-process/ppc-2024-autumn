#include "mpi/gnitienko_k_bellman-ford-algorithm/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

void gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq::toCRS(const int* input_matrix) {
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

bool gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq::pre_processing() {
  internal_order_test();
  auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);

  toCRS(input_matrix);

  shortest_paths.resize(V, INF);
  shortest_paths[0] = 0;
  return true;
}

bool gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq::validation() {
  internal_order_test();
  auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
  V = taskData->inputs_count[0];
  size_t j = 0;
  for (size_t i = 0; i < V; i++) {
    if (input_matrix[i * V + j] != 0) return false;
    j++;
  }
  return taskData->inputs_count[0] == taskData->outputs_count[0] && taskData->inputs_count[0] != 0;
}

bool gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq::Iteration(std::vector<int>& paths) {
  bool changed = false;
  for (size_t i = 0; i < V; i++) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
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

bool gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq::check_negative_cycle() {
  for (size_t i = 0; i < V; ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      int v = columns[j];
      int weight = values[j];
      if (shortest_paths[i] != INF && shortest_paths[i] + weight < shortest_paths[v]) {
        std::cerr << "Negative cycle detected in seq!" << std::endl;
        return true;
      }
    }
  }
  return false;
}

bool gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq::run() {
  internal_order_test();

  bool changed = false;
  for (size_t i = 0; i < V - 1; i++) {
    changed = Iteration(shortest_paths);
    if (!changed) break;
  }

  for (size_t i = 0; i < V; i++) {
    if (shortest_paths[i] == INF) shortest_paths[i] = 0;
  }

  return !check_negative_cycle();
}

bool gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgSeq::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < V; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = shortest_paths[i];
  }
  return true;
}

// PARALLEL
//----------------------------------------------------------------------------

void gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI::toCRS(const int* input_matrix) {
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

  values_size = values.size();
  columns_size = columns.size();
  row_ptr_size = row_ptr.size();
}

bool gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI::check_negative_cycle() {
  for (size_t i = 0; i < V; ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      int v = columns[j];
      int weight = values[j];
      if (shortest_paths[i] != INF && shortest_paths[i] + weight < shortest_paths[v]) {
        std::cerr << "Negative cycle detected in mpi!" << std::endl;
        return true;
      }
    }
  }
  return false;
}

bool gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI::Iteration(std::vector<int>& paths) {
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
  boost::mpi::reduce(world, start_paths.data(), V, reduced_paths.data(), boost::mpi::minimum<int>(), 0);

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

bool gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);

    toCRS(input_matrix);
  }
  return true;
}

bool gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
    V = taskData->inputs_count[0];
    size_t j = 0;
    for (size_t i = 0; i < V; i++) {
      if (input_matrix[i * V + j] != 0) return false;
      j++;
    }
    return taskData->inputs_count[0] == taskData->outputs_count[0] && taskData->inputs_count[0] != 0;
  }
  return true;
}

bool gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI::run() {
  internal_order_test();
  boost::mpi::broadcast(world, V, 0);
  boost::mpi::broadcast(world, values_size, 0);
  boost::mpi::broadcast(world, columns_size, 0);
  boost::mpi::broadcast(world, row_ptr_size, 0);
  values.resize(values_size);
  columns.resize(columns_size);
  row_ptr.resize(row_ptr_size);
  boost::mpi::broadcast(world, values.data(), values.size(), 0);
  boost::mpi::broadcast(world, columns.data(), columns.size(), 0);
  boost::mpi::broadcast(world, row_ptr.data(), row_ptr.size(), 0);

  shortest_paths.resize(V, INF);
  shortest_paths[0] = 0;

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

bool gnitienko_k_bellman_ford_algorithm_mpi::BellmanFordAlgMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < V; ++i) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = shortest_paths[i];
    }
  }
  return true;
}
