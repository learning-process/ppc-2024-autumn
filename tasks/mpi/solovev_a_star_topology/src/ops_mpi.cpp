#include "mpi/solovev_a_star_topology/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace solovev_a_star_topology_mpi {

bool solovev_a_star_topology_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_.resize(taskData->inputs_count[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  }
  res = std::vector<int>(input_.size(), 0);
  return true;
}

bool solovev_a_star_topology_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if ((taskData->inputs_count[0] == taskData->outputs_count[0]) && (taskData->inputs_count[0] > 0) &&
        (taskData->outputs_count[0] > 0)) {
      return true;
    }
    return false;
  }
  return true;
}

bool solovev_a_star_topology_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 1; i < world.size(); ++i) {
      world.send(i, 0, input_);
    }
  } else {
    world.recv(0, 0, input_);
    world.send(0, 0, input_);
  }
  if (world.rank() == 0) {
    for (int i = 1; i < world.size(); ++i) {
      world.recv(i, 0, res);
    }
  }
  return true;
}

bool solovev_a_star_topology_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  world.barrier();
  if (world.rank() == 0) {
    std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
}  // namespace solovev_a_star_topology_mpi
