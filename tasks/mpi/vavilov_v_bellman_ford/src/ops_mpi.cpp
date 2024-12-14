#include "mpi/vavilov_v_bellman_ford/include/ops_mpi.hpp"

bool vavilov_v_bellman_ford_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  vertices_ = taskData->inputs_count[0];
  edges_count_ = taskData->inputs_count[1];
  source_ = taskData->inputs_count[2];

  row_offsets_.resize(vertices_ + 1);
  col_indices_.resize(edges_count_);
  weights_.resize(edges_count_);
  
  std::copy(reinterpret_cast<int*>(taskData->inputs[0]),
            reinterpret_cast<int*>(taskData->inputs[0]) + row_offsets_.size(),
            row_offsets_.begin());
  std::copy(reinterpret_cast<int*>(taskData->inputs[1]),
            reinterpret_cast<int*>(taskData->inputs[1]) + col_indices_.size(),
            col_indices_.begin());
  std::copy(reinterpret_cast<int*>(taskData->inputs[2]),
            reinterpret_cast<int*>(taskData->inputs[2]) + weights_.size(),
            weights_.begin());

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
    for (int u = 0; u < vertices_; ++u) {
      for (int j = row_offsets_[u]; j < row_offsets_[u + 1]; ++j) {
        int v = col_indices_[j];
        int weight = weights_[j];
        if (distances_[u] != INT_MAX && distances_[u] + weight < distances_[v]) {
          distances_[v] = distances_[u] + weight;
        }
      }
    }
  }

  for (int u = 0; u < vertices_; ++u) {
    for (int j = row_offsets_[u]; j < row_offsets_[u + 1]; ++j) {
      int v = col_indices_[j];
      int weight = weights_[j];
      if (distances_[u] != INT_MAX && distances_[u] + weight < distances_[v]) {
        return false;  // Negative weight cycle detected
      }
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
  vertices_ = taskData->inputs_count[0];
  edges_count_ = taskData->inputs_count[1];
  source_ = taskData->inputs_count[2];

 std::copy(reinterpret_cast<int*>(taskData->inputs[0]),
            reinterpret_cast<int*>(taskData->inputs[0]) + row_offsets_.size(),
            row_offsets_.begin());
  std::copy(reinterpret_cast<int*>(taskData->inputs[1]),
            reinterpret_cast<int*>(taskData->inputs[1]) + col_indices_.size(),
            col_indices_.begin());
  std::copy(reinterpret_cast<int*>(taskData->inputs[2]),
            reinterpret_cast<int*>(taskData->inputs[2]) + weights_.size(),
            weights_.begin());

  boost::mpi::broadcast(world, row_offsets_, 0);
  boost::mpi::broadcast(world, col_indices_, 0);
  boost::mpi::broadcast(world, weights_, 0);

  distances_.resize(vertices_, INT_MAX);
  distances_[source_] = 0;

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

  int local_start = world.rank() * (vertices_ / world.size());
  int local_end = (world.rank() == world.size() - 1) ? vertices_ : local_start + (vertices_ / world.size());

  for (int i = 0; i < vertices_ - 1; ++i) {
    std::vector<int> local_distances = distances_;
    bool local_changed = false;

    for (int u = local_start; u < local_end; ++u) {
      for (int j = row_offsets_[u]; j < row_offsets_[u + 1]; ++j) {
        int v = col_indices_[j];
        int weight = weights_[j];
        if (distances_[u] != INT_MAX && distances_[u] + weight < local_distances[v]) {
          local_distances[v] = distances_[u] + weight;
          local_changed = true;
        }
      }
    }

    boost::mpi::all_reduce(world, local_distances.data(), vertices_, distances_.data(),
                           [](int a, int b) {
                             return (a == INT_MAX) ? b : (b == INT_MAX) ? a : std::min(a, b);
                           });

    bool global_changed = boost::mpi::all_reduce(world, local_changed, std::logical_or<bool>());
    if (!global_changed) break;
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
