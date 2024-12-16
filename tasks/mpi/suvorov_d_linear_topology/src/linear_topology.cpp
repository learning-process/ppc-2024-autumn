// Copyright 2023 Nesterov Alexander
#include "mpi/suvorov_d_linear_topology/include/linear_topology.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

bool suvorov_d_linear_topology_mpi::MPILinearTopology::pre_processing() {
  internal_order_test();

  if (world.size() == 1) return true;

  std::uint32_t data_size = 0;
  if (world.rank() == 0) {
    rank_order_.resize(0);
    data_size = taskData->inputs_count[0];
    // Init data vector in proc with rank 0
    int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    local_data_.resize(data_size);
    std::copy(tmp_ptr, tmp_ptr + data_size, local_data_.begin());
  }

  return true;
}

bool suvorov_d_linear_topology_mpi::MPILinearTopology::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] > 0;
  }
  return true;
}

bool suvorov_d_linear_topology_mpi::MPILinearTopology::run() {
  internal_order_test();

  // Resizing all vectors
  std::uint32_t data_size = 0;
  std::vector<int> verific_data_;
  if (world.rank() == 0) {
    data_size = taskData->inputs_count[0];
  }
  boost::mpi::broadcast(world, data_size, 0);
  if (world.rank() == world.size() - 1) {
    verific_data_.resize(data_size);
  }
  local_data_.resize(data_size);

  if (world.size() == 1) return true;

  if (world.rank() == 0) {
    rank_order_.push_back(0);
    world.send(1, 0, local_data_);
    world.send(1, 1, rank_order_);
    world.send(world.size() - 1, 2, local_data_);
  } else {
    world.recv(world.rank() - 1, 0, local_data_);
    world.recv(world.rank() - 1, 1, rank_order_);

    if (world.rank() == world.size() - 1) {
      world.recv(0, 2, verific_data_);
    }

    rank_order_.push_back(static_cast<size_t>(world.rank()));

    if (world.rank() != world.size() - 1) {
      world.send(world.rank() + 1, 0, local_data_);
      world.send(world.rank() + 1, 1, rank_order_);
    }
  }

  // Checking result of sending
  if (world.rank() == world.size() - 1) {
    bool order_is_ok = true;
    for (size_t i = 0; i < rank_order_.size(); i++) {
      if (rank_order_[i] != i) {
        order_is_ok = false;
        break;
      }
    }
    order_is_ok = rank_order_.size() == static_cast<size_t>(world.size()) ? order_is_ok : false;

    if (local_data_ == verific_data_ && order_is_ok) {
      world.send(0, 3, true);
    } else {
      world.send(0, 3, false);
    }
  }
  if (world.rank() == 0) {
    world.recv(world.size() - 1, 3, result_);
  }

  return true;
}

bool suvorov_d_linear_topology_mpi::MPILinearTopology::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    bool* output_data_ptr = reinterpret_cast<bool*>(taskData->outputs[0]);
    output_data_ptr[0] = world.size() == 1 ? true : result_;
  }

  return true;
}
