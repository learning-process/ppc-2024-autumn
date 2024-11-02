// Copyright 2023 Nesterov Alexander
#include "mpi/titov_s_vector_sum/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> titov_s_vector_sum_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool titov_s_vector_sum_mpi::MPIVectorSumSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init value for output
  res = 0;
  return true;
}

bool titov_s_vector_sum_mpi::MPIVectorSumSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool titov_s_vector_sum_mpi::MPIVectorSumSequential::run() {
  internal_order_test();
  res = std::accumulate(input_.begin(), input_.end(), 0);
  return true;
}

bool titov_s_vector_sum_mpi::MPIVectorSumSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool titov_s_vector_sum_mpi::MPIVectorSumParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  unsigned int remainder = 0;

  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    remainder = taskData->inputs_count[0] % world.size();
  }

  broadcast(world, delta, 0);
  broadcast(world, remainder, 0);

  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);

    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }

    for (int proc = 1; proc < world.size(); proc++) {
      unsigned int send_size = (proc == world.size() - 1) ? delta + remainder : delta;
      world.send(proc, 0, input_.data() + proc * delta, send_size);
    }
  }
  local_input_ = std::vector<int>((world.rank() == world.size() - 1) ? delta + remainder : delta);

  if (world.rank() != 0) {
    unsigned int recv_size = (world.rank() == world.size() - 1) ? delta + remainder : delta;
    world.recv(0, 0, local_input_.data(), recv_size);
  } else {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  }

  res = 0;
  return true;
}

bool titov_s_vector_sum_mpi::MPIVectorSumParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool titov_s_vector_sum_mpi::MPIVectorSumParallel::run() {
  internal_order_test();
  int local_res;
  local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  reduce(world, local_res, res, std::plus(), 0);
  return true;
}

bool titov_s_vector_sum_mpi::MPIVectorSumParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
