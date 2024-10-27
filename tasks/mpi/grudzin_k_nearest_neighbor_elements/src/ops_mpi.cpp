// Copyright 2023 Nesterov Alexander
#include "mpi/grudzin_k_nearest_neighbor_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> grudzin_k_nearest_neighbor_elements_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<std::pair<int, int>>(taskData->inputs_count[0] - 1);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0] - 1; i++) {
    input_[i] = {abs(tmp_ptr[i] - tmp_ptr[i + 1]), i};
  }
  // Init value for output
  res = {INT_MAX, -1};
  return true;
}

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); ++i) {
    res = std::min(res, input_[i]);
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
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = (taskData->inputs_count[0] - 1) / world.size();
    if ((taskData->inputs_count[0] - 1) % world.size()) delta++;
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<std::pair<int, int>>(world.size() * delta);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned int i = 0; i < taskData->inputs_count[0] - 1; i++) {
      input_[i] = {abs(tmp_ptr[i + 1] - tmp_ptr[i]), i};
    }
    for (size_t i = taskData->inputs_count[0] - 1; i < input_.size(); ++i) {
      input_[i] = {INT_MAX, -1};
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }
  local_input_ = std::vector<std::pair<int, int>>(delta, {INT_MAX, -1});
  if (world.rank() == 0) {
    local_input_ = std::vector<std::pair<int, int>>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  // Init value for output
  res = {INT_MAX, -1};
  return true;
}

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool grudzin_k_nearest_neighbor_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  std::pair<int, int> local_ans_ = {INT_MAX, -1};
  for (size_t i = 0; i < local_input_.size(); ++i) {
    local_ans_ = std::min(local_ans_, local_input_[i]);
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
