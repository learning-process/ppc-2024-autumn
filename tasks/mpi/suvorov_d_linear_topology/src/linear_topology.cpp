// Copyright 2023 Nesterov Alexander
#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "mpi\suvorov_d_linear_topology\include\linear_topology.hpp"

bool suvorov_d_linear_topology_mpi::MPILinearTopology::pre_processing() {
  internal_order_test();

  if (world.size() == 1) return true;

  std::uint32_t data_size = 0;
  if (world.rank() == 0) {
    data_size = taskData->inputs_count[0];
    boost::mpi::broadcast(world, data_size, 0);

    // Init data vector in proc with rank 0
    int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    local_data_.resize(data_size);
    std::copy(tmp_ptr, tmp_ptr + data_size, local_data_.begin());
  }
  if (world.rank() == world.size() - 1) {
    verific_data_.resize(data_size);
  }

  local_data_.resize(data_size);

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

  if (world.size() == 1) return true;

  if (world.rank() == 0) {
    rank_order_.push_back(0);
    world.send(1, 0, local_data_);
    world.send(1, 1, rank_order_);

    world.send(world.size() - 1, 1, local_data_);
  } else {
    world.recv(world.rank() - 1, 0, local_data_);
    world.recv(world.rank() - 1, 1, rank_order_);

    rank_order_.push_back(world.rank());

    if (world.rank() != world.size() - 1) {
      world.send(world.rank() + 1, 0, local_data_);
      world.send(world.rank() + 1, 1, rank_order_);
    } else {
      world.recv(0, 1, verific_data_);
    }
  }

  return true;
}

bool suvorov_d_linear_topology_mpi::MPILinearTopology::post_processing() {
  internal_order_test();

  if (world.size() == 1) {
    int* output_data_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    output_data_ptr[0] = true;
    return true;
  }

  if (world.rank() == world.size() - 1) {
    bool order_is_ok = true;
    for (int i = 0; i < rank_order_.size(); i++) {
      if (rank_order_[i] != i) order_is_ok = false;
    }
    order_is_ok = rank_order_.size() == world.size() ? order_is_ok : false;

    if (local_data_ == verific_data_ && order_is_ok) {
      world.send(0, 0, true);
    } else {
      world.send(0, 0, false);
    }
  }
  if (world.rank() == 0) {
    bool data_correct;
    world.recv(world.size() - 1, 0, data_correct);
    int* output_data_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    output_data_ptr[0] = data_correct;
  }
  return true;
}
