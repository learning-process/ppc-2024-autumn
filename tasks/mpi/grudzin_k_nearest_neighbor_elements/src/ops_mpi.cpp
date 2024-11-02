// Copyright 2023 Nesterov Alexander
#include "mpi/grudzin_k_nearest_neighbor_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <random>
#include <vector>

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  // Init value for output
  res = {INT_MAX, -1};
  return true;
}

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 1 && taskData->outputs_count[0] == 1;
}

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size() - 1; ++i) {
    std::pair<int, int> tmp = {abs(input_[i] - input_[i + 1]), i};
    res = std::min(res, tmp);
  }
  return true;
}

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res.second;
  return true;
}

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  // Init value for output
  res = {INT_MAX, -1};
  return true;
}

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] > 1 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  size_t delta = 0;
  if (world.rank() == 0) {
    delta = (taskData->inputs_count[0]) / world.size();
    size = taskData->inputs_count[0];
    if (taskData->inputs_count[0] % world.size() > 0u) delta++;
  }
  broadcast(world, delta, 0);
  broadcast(world, size, 0);

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(size, 0);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
    for (int proc = 1; proc < world.size(); proc++) {
      if (proc * delta < size) {
        size_t dpo = delta + 1;
        size_t pack_size = std::min(dpo, size - proc * delta);
        world.send(proc, 0, input_.data() + proc * delta, pack_size);
      }
    }
  }

  local_input_ = std::vector<int>(delta + 1);
  start = world.rank() * delta;
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta + 1);
  } else if (world.rank() * delta < size) {
    size_t dpo = delta + 1;
    size_t pack_size = std::min(dpo, size - world.rank() * delta);
    world.recv(0, 0, local_input_.data(), pack_size);
  }

  std::pair<int, int> local_ans_ = {INT_MAX, -1};
  for (size_t i = 0; i < local_input_.size() - 1 && (i + start) < size - 1; ++i) {
    std::pair<int, int> tmp = {abs(local_input_[i] - local_input_[i + 1]), i + start};
    local_ans_ = std::min(local_ans_, tmp);
  }
  reduce(world, local_ans_, res, boost::mpi::minimum<std::pair<int, int>>(), 0);
  return true;
}

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res.second;
  }
  return true;
}
