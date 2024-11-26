#include "mpi/bessonov_e_star_topology/include/ops_mpi.hpp"

bool bessonov_e_star_topology_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int num_elements = taskData->inputs_count[0];
    int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
    input_.assign(input_data, input_data + num_elements);

    traversal_order_.clear();
  }

  return true;
}

bool bessonov_e_star_topology_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() < 2) {
    return false;
  }

  if (world.rank() == 0) {
    return (taskData->inputs_count.size() >= 1) && (taskData->inputs_count[0] > 0) && (!taskData->inputs.empty()) &&
           (taskData->inputs[0] != nullptr);
  }

  return true;
}

bool bessonov_e_star_topology_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  traversal_order_.push_back(0);

  if (world.rank() == 0) {
    for (int dest = 1; dest < world.size(); ++dest) {
      world.send(dest, 0, input_);

      traversal_order_.push_back(dest);

      std::vector<int> received_data;
      world.recv(dest, 0, received_data);

      traversal_order_.push_back(0);
    }
  } else {
    world.recv(0, 0, input_);
    world.send(0, 0, input_);
  }

  return true;
}

bool bessonov_e_star_topology_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(input_.begin(), input_.end(), reinterpret_cast<int*>(taskData->outputs[0]));

    std::copy(traversal_order_.begin(), traversal_order_.end(), reinterpret_cast<int*>(taskData->outputs[1]));
  }

  return true;
}