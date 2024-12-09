// Copyright 2023 Nesterov Alexander
#include "mpi/korovin_n_line_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool korovin_n_line_topology_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int root = taskData->inputs_count[0];
  int num_of_elements = taskData->inputs_count[2];

  if (world.rank() == root) {
    auto* input_buffer = reinterpret_cast<int*>(taskData->inputs[0]);
    data_.assign(input_buffer, input_buffer + num_of_elements);
    trajectory_.clear();
    trajectory_.emplace_back(world.rank());
  }

  return true;
}

bool korovin_n_line_topology_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 3) return false;

  int root = taskData->inputs_count[0];
  int dst = taskData->inputs_count[1];
  int num_of_elements = taskData->inputs_count[2];

  return (root >= 0 && root < world.size() && dst >= 0 && dst < world.size() && num_of_elements > 0) &&
         ((world.rank() != root) || ((!taskData->inputs.empty()) && (taskData->inputs[0] != nullptr))) &&
         ((world.rank() != dst) || ((!taskData->outputs.empty()) && (taskData->outputs[0] != nullptr)));
}

bool korovin_n_line_topology_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int root = taskData->inputs_count[0];
  int dst = taskData->inputs_count[1];

  if (root == dst) {
    return true;
  }

  if (world.rank() < root || world.rank() > dst) {
    return true;
  }
  if (world.rank() == root) {
    world.send(world.rank() + 1, 0, data_);
    world.send(world.rank() + 1, 1, trajectory_);
  } else {
    world.recv(world.rank() - 1, 0, data_);
    world.recv(world.rank() - 1, 1, trajectory_);
    trajectory_.emplace_back(world.rank());
    if (world.rank() < dst) {
      world.send(world.rank() + 1, 0, data_);
      world.send(world.rank() + 1, 1, trajectory_);
    }
  }

  return true;
}

bool korovin_n_line_topology_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  int dst = taskData->inputs_count[1];

  if (world.rank() == dst) {
    auto* data_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(data_.begin(), data_.end(), data_ptr);
    auto* trajectory_ptr = reinterpret_cast<int*>(taskData->outputs[1]);
    std::copy(trajectory_.begin(), trajectory_.end(), trajectory_ptr);
  }

  return true;
}
