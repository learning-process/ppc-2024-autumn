// Copyright 2023 Nesterov Alexander
#include "mpi/yasakova_t_reduce/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

bool yasakova_t_reduce::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    columns = taskData->inputs_count[1];

    input_data_ = std::vector<int>(rows * columns);

    for (int i = 0; i < rows; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      for (int j = 0; j < columns; j++) {
        input_data_[i * columns + j] = tmp_ptr[j];
      }
    }
  }
  result_ = INT_MAX;
  return true;
}

bool yasakova_t_reduce::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1 && !(taskData->inputs.empty());
  }
  return true;
}

bool yasakova_t_reduce::TestMPITaskParallel::run() {
  internal_order_test();

  unsigned int delta = 0;
  unsigned int extra = 0;

  if (world.rank() == 0) {
    delta = rows * columns / world.size();
    extra = rows * columns % world.size();
  }

  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_data_.data() + delta * proc + extra, delta);
    }
  }

  local_input_data_ = std::vector<int>(delta);

  if (world.rank() == 0) {
    local_input_data_ = std::vector<int>(input_data_.begin(), input_data_.begin() + delta + extra);
  } else {
    world.recv(0, 0, local_input_data_.data(), delta);
  }

  int local_min = INT_MAX;
  if (!local_input_data_.empty()) {
    local_min = *std::min_element(local_input_data_.begin(), local_input_data_.end());
  }
  reduce(world, local_min, result_, boost::mpi::minimum<int>(), 0);
  return true;
}

bool yasakova_t_reduce::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = result_;
  }
  return true;
}