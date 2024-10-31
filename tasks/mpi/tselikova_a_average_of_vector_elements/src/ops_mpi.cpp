// Copyright 2023 Nesterov Alexander
#include "mpi/tselikova_a_average_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;


bool tselikova_a_average_of_vector_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  int* tmp = reinterpret_cast<int*>(taskData->inputs[0]);
  input_ = std::vector<int>(taskData->inputs_count[0]);
  for (int i = 0; i < (int)taskData->inputs_count[0]; i++) {
    input_[i] = tmp[i];
    // std::cout << input_[i] << " ";
  }
  // std::cout << std::endl;
  // std::copy(&tmp[0], &taskData->inputs_count[0], input_);
  res = 0;
  return true;
}

bool tselikova_a_average_of_vector_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1;
}

bool tselikova_a_average_of_vector_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  int sum = 0;
  for (int i = 0; i < input_.size(); i++) {
    sum += input_[i];
    // std::cout << sum << " ";
  }
  // std::cout << std::endl;
  // std::cout << input_.size();
  res = sum / input_.size();
  return true;
}

bool tselikova_a_average_of_vector_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::cout << res << std::endl;
  reinterpret_cast<float*>(taskData->outputs[0])[0] = res;
  return true;
}


bool tselikova_a_average_of_vector_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }
  local_input_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  // Init value for output
  res = 0;
  return true;
}

bool tselikova_a_average_of_vector_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] >= 0;
  }
  return true;
}

bool tselikova_a_average_of_vector_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int local_sum = 0;
  for (unsigned int i = 0; i < local_input_.size(); i++) {
    local_sum += local_input_[i];
  }
  reduce(world, local_sum, sum_, std::plus(), 0);
  return true;
}

bool tselikova_a_average_of_vector_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    res = sum_ / world.size();
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}

