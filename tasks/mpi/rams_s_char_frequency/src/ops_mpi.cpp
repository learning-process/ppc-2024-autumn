// Copyright 2023 Nesterov Alexander
#include "mpi/rams_s_char_frequency/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "boost/mpi/collectives/scatterv.hpp"

using namespace std::chrono_literals;

bool rams_s_char_frequency_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  target_ = *reinterpret_cast<char*>(taskData->inputs[1]);
  res = 0;
  return true;
}

bool rams_s_char_frequency_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] == 1 && taskData->outputs_count[0] == 1;
}

bool rams_s_char_frequency_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = std::count(input_.begin(), input_.end(), target_);
  return true;
}

bool rams_s_char_frequency_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool rams_s_char_frequency_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int total = 0;
  if (world.rank() == 0) {
    total = taskData->inputs_count[0];
  }

  if (world.rank() == 0) {
    input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
    target_ = *reinterpret_cast<char*>(taskData->inputs[1]);
  }

  res = 0;
  local_res = 0;
  return true;
}

bool rams_s_char_frequency_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] == 1 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool rams_s_char_frequency_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  broadcast(world, total, 0);
  broadcast(world, target_, 0);
  unsigned int delta = total / world.size();
  unsigned int overflow = total % world.size();

  std::vector<int> sizes(world.size(), delta);
  sizes[world.size() - 1] = delta + overflow;
  std::vector<int> displs(world.size());
  for (int i = 1; i < world.size(); i++) {
    displs[i] = displs[i - 1] + delta;
  }

  unsigned int local_delta = sizes[world.rank()];
  local_input_.resize(local_delta);

  boost::mpi::scatterv(world, input_.data(), sizes, displs, local_input_.data(), local_delta, 0);
  local_res = std::count(local_input_.begin(), local_input_.end(), target_);
  reduce(world, local_res, res, std::plus(), 0);
  return true;
}

bool rams_s_char_frequency_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
