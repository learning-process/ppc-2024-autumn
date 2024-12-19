// Copyright 2023 Nesterov Alexander
#include "mpi/ermilova_d_reduceMPI/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

bool ermilova_d_reduceMPI_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];

    input_ = std::vector<int>(rows * cols);

    for (int i = 0; i < rows; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      for (int j = 0; j < cols; j++) {
        input_[i * cols + j] = tmp_ptr[j];
      }
    }
  }
  res = INT_MAX;
  return true;
}

bool ermilova_d_reduceMPI_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1 && !(taskData->inputs.empty());
  }
  return true;
}

bool ermilova_d_reduceMPI_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  unsigned int delta = 0;
  unsigned int extra = 0;

  if (world.rank() == 0) {
    delta = rows * cols / world.size();
    extra = rows * cols % world.size();
  }

  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + delta * proc + extra, delta);
    }
  }

  local_input_ = std::vector<int>(delta);

  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta + extra);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }

  int local_min = INT_MAX;
  if (!local_input_.empty()) {
    local_min = *std::min_element(local_input_.begin(), local_input_.end());
  }
  reduce(world, local_min, res, boost::mpi::minimum<int>(), 0);
  return true;
}

bool ermilova_d_reduceMPI_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
