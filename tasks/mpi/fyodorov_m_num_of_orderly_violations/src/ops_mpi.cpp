// Copyright 2023 Nesterov Alexander
#include "mpi/fyodorov_m_num_of_orderly_violations/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> fyodorov_m_num_of_orderly_violations_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }

  violations_count = 0;
  return true;
}

bool fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 1; i < input_.size(); ++i) {
    if (input_[i] < input_[i - 1]) {
      ++violations_count;
    }
  }
  return true;
}

bool fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = violations_count;
  return true;
}

bool fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  }

  violations_count = 0;
  return true;
}

bool fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  unsigned int res = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    res = taskData->inputs_count[0] % world.size();
  }
  broadcast(world, delta, 0);
  broadcast(world, res, 0);

  if (world.rank() == 0) {
    // Init vectors
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta + res - 1, delta + 1);
    }
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta + res);
  } else {
    local_input_ = std::vector<int>(delta + 1);
    world.recv(0, 0, local_input_.data(), delta + 1);
  }

  int local_violations = 0;
  for (size_t i = 1; i < local_input_.size(); ++i) {
    if (local_input_[i] < local_input_[i - 1]) {
      ++local_violations;
    }
  }

  reduce(world, local_violations, violations_count, std::plus(), 0);
  return true;
}

bool fyodorov_m_num_of_orderly_violations_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = violations_count;
  }
  return true;
}
