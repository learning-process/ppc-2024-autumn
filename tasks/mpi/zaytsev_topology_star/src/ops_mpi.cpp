// Copyright 2023 Nesterov Alexander
#include "mpi/zaytsev_topology_star/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <vector>

bool zaytsev_topology_star::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
    trajectory.clear();
  }
  return true;
}

bool zaytsev_topology_star::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count[0] < 0) {
      return false;
    }
  }

  return true;
}

bool zaytsev_topology_star::TestMPITaskParallel::run() {
  internal_order_test();

  if (world.rank() == 0) {
    trajectory.push_back(0);

    for (int i = 1; i < world.size(); ++i) {
      world.send(i, 0, input_);
      world.send(i, 1, trajectory);

      std::vector<int> received_trajectory;
      world.recv(i, 0, received_trajectory);

      trajectory = received_trajectory;
      trajectory.push_back(0);
    }

  } else {
    std::vector<int> received_data;
    std::vector<int> received_trajectory;

    world.recv(0, 0, received_data);
    world.recv(0, 1, received_trajectory);

    received_trajectory.push_back(world.rank());

    world.send(0, 0, received_trajectory);
  }

  return true;
}

bool zaytsev_topology_star::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(input_.begin(), input_.end(), output_ptr);
    auto* trajectory_ptr = output_ptr + input_.size();
    std::copy(trajectory.begin(), trajectory.end(), trajectory_ptr);
    taskData->outputs_count[0] = input_.size() + trajectory.size();
  }

  return true;
}