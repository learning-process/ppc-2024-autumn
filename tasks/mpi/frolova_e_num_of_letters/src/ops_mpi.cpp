// Copyright 2023 Nesterov Alexander
#include "mpi/frolova_e_num_of_letters/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::string frolova_e_num_of_letters_mpi::GenStr(int n) {
  std::string str = "test";
  std::string result;
  result.resize(n);

  int i = 0;
  size_t j = 0;

  while (i < n) {
    result[i] = str[j];
    j++;
    i++;
    if (j >= str.size()) {
      j = 0;
    }
  }
  return result;
}

bool frolova_e_num_of_letters_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  // Init value for output
  res = 0;
  return true;
}

bool frolova_e_num_of_letters_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool frolova_e_num_of_letters_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (char c : input_) {
    if (isalpha(c)) res++;
  }
  return true;
}

bool frolova_e_num_of_letters_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool frolova_e_num_of_letters_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]); 
  } 

  res = 0;
  return true;
}

bool frolova_e_num_of_letters_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
  }
  return true;
}

bool frolova_e_num_of_letters_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int delta = 0;

  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + (proc - 1) * delta, delta);
    }
    local_input_ = std::string(input_.begin() + (world.size() - 1) * delta, input_.end());
  } else {
    local_input_.resize(delta);
    world.recv(0, 0, local_input_.data(), delta);
  }
  int local_res = 0;
  for (char c : local_input_) {
    if (isalpha(c)) {
      local_res++;
    }
  }

  reduce(world, local_res, res, std::plus(), 0);
  return true;
}

bool frolova_e_num_of_letters_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}