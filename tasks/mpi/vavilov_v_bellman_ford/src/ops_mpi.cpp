#include "mpi/vavilov_v_bellman_ford/include/ops_mpi.hpp"

bool vavilov_v_bellman_ford_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  vertices_ = taskData->inputs_count[0];
  edges_count_ = taskData->inputs_count[1];
  source_ = taskData->inputs_count[2];
  int* edges_data = reinterpret_cast<int*>(taskData->inputs[0]);

  for (int i = 0; i < edges_count_; ++i) {
    edges_.push_back({edges_data[i * 3], edges_data[i * 3 + 1], edges_data[i * 3 + 2]});
  }

  distances_.resize(vertices_, INT_MAX);
  distances_[source_] = 0;

  return true;
}

bool vavilov_v_bellman_ford_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return (!taskData->inputs.empty());
}

bool vavilov_v_bellman_ford_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (int i = 1; i < vertices_; ++i) {
    for (const auto& edge : edges_) {
      if (distances_[edge.src] != INT_MAX && distances_[edge.src] + edge.weight < distances_[edge.dest]) {
        distances_[edge.dest] = distances_[edge.src] + edge.weight;
      }
    }
  }

  for (const auto& edge : edges_) {
    if (distances_[edge.src] != INT_MAX && distances_[edge.src] + edge.weight < distances_[edge.dest]) {
      return false;  // Negative weight cycle detected
    }
  }

  return true;
}

bool vavilov_v_bellman_ford_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  std::copy(distances_.begin(), distances_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

bool vavilov_v_bellman_ford_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  return true;
}

bool vavilov_v_bellman_ford_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return (!taskData->inputs.empty());
  }
  return true;
}

bool vavilov_v_bellman_ford_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int vertices = taskData->inputs_count[0];
  int edges_count = taskData->inputs_count[1];
  int source = taskData->inputs_count[2];

  if (world.rank() == 0) {
    edges_.resize(edges_count);
    int* edges_data = reinterpret_cast<int*>(taskData->inputs[0]);
    for (int i = 0; i < edges_count; ++i) {
      edges_[i] = {edges_data[i * 3], edges_data[i * 3 + 1], edges_data[i * 3 + 2]};
    }
  }

  boost::mpi::broadcast(world, edges_, 0);

  distances_.resize(vertices, INT_MAX);
  if (world.rank() == 0) {
    distances_[source] = 0;
  }

  boost::mpi::broadcast(world, distances_, 0);

  for (int i = 1; i < vertices; ++i) {
    std::vector<int> local_distances = distances_;

    for (int j = world.rank(); j < edges_count; j += world.size()) {
      const auto& edge = edges_[j];
      if (distances_[edge.src] != INT_MAX && distances_[edge.src] + edge.weight < local_distances[edge.dest]) {
        local_distances[edge.dest] = distances_[edge.src] + edge.weight;
      }
    }

    boost::mpi::all_reduce(world, local_distances.data(), vertices, distances_.data(), boost::mpi::minimum<int>());
  }

  bool has_negative_cycle = false;
  for (int j = world.rank(); j < edges_count; j += world.size()) {
    const auto& edge = edges_[j];
    if (distances_[edge.src] != INT_MAX && distances_[edge.src] + edge.weight < distances_[edge.dest]) {
      has_negative_cycle = true;
      break;
    }
  }

  has_negative_cycle = boost::mpi::all_reduce(world, has_negative_cycle, std::logical_or<bool>());
  if (has_negative_cycle) {
    return false;
  }

  return true;
}

bool vavilov_v_bellman_ford_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(distances_.begin(), distances_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
