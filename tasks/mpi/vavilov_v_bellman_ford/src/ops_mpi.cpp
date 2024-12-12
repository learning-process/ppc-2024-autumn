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
  if (world.rank() == 0) {
    vertices_ = taskData->inputs_count[0];
    edges_count_ = taskData->inputs_count[1];
    source_ = taskData->inputs_count[2];

    int* edges_data = reinterpret_cast<int*>(taskData->inputs[0]);
    for (int i = 0; i < edges_count_; ++i) {
      edges_.push_back({edges_data[i * 3], edges_data[i * 3 + 1], edges_data[i * 3 + 2]});
    }
    distances_.resize(vertices_, INT_MAX);
    distances_[source_] = 0;
  }

  boost::mpi::broadcast(world, vertices_, 0);
  boost::mpi::broadcast(world, edges_, 0);
  boost::mpi::broadcast(world, distances_, 0);

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

  int local_start = world.rank() * (edges_count_ / world.size()) + std::min(world.rank(), edges_count_ % world.size());
  int local_end = local_start + (edges_count_ / world.size()) + (world.rank() < edges_count_ % world.size() ? 1 : 0);

  for (int i = 0; i < vertices_ - 1; ++i) {
    std::vector<int> local_distances = distances_;
    bool local_changed = false;

    for (int j = local_start; j < local_end; ++j) {
      const auto& edge = edges_[j];
      if (distances_[edge.src] != INT_MAX && distances_[edge.src] + edge.weight < local_distances[edge.dest]) {
        local_distances[edge.dest] = distances_[edge.src] + edge.weight;
        local_changed = true;
      }
    }

    boost::mpi::all_reduce(world, local_distances.data(), vertices_, distances_.data(),
                           [](int a, int b) {
                             if (a == INT_MAX) return b;
                             if (b == INT_MAX) return a;
                             return std::min(a, b); });

    bool global_changed = boost::mpi::all_reduce(world, local_changed, std::logical_or<bool>());
    if (!global_changed) break;
  }

  bool has_negative_cycle = false;
  for (int j = local_start; j < local_end; ++j) {
    const auto& edge = edges_[j];
    if (distances_[edge.src] != INT_MAX && distances_[edge.src] + edge.weight < distances_[edge.dest]) {
      has_negative_cycle = true;
      break;
    }
  }

  has_negative_cycle = boost::mpi::all_reduce(world, has_negative_cycle, std::logical_or<bool>());
  return !has_negative_cycle;
}

bool vavilov_v_bellman_ford_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(distances_.begin(), distances_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
