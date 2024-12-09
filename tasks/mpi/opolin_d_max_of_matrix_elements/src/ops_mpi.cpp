// Copyright 2023 Nesterov Alexander
#include "mpi/opolin_d_max_of_matrix_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace std::chrono_literals;

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init matrix
  unsigned int rows = taskData->inputs_count[0];
  unsigned int cols = taskData->inputs_count[1];
  input_ = std::vector<std::vector<int>>(rows, std::vector<int>(cols));
  for (unsigned int i = 0; i < rows; i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < cols; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  // Init value for output
  res = std::numeric_limits<int32_t>::min();
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check non empty input data
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); i++) {
    for (size_t j = 0; j < input_[i].size(); j++) {
      if (input_[i][j] > res) {
        res = input_[i][j];
      }
    }
  }
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[0] == 0 || taskData->inputs_count[1] == 0) {
      return false;
    }
    // Init vectors
    unsigned int rows = taskData->inputs_count[0];
    unsigned int cols = taskData->inputs_count[1];
    unsigned int total_elements = rows * cols;
    input_ = std::vector<int>(total_elements);
    // Init input vector
    for (unsigned int i = 0; i < rows; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      for (unsigned int j = 0; j < cols; j++) {
        input_[i * cols + j] = tmp_ptr[j];
      }
    }
  }
  // Init value for output
  res = std::numeric_limits<int32_t>::min();
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check non empty input data
    return taskData->outputs_count[0] == 1 && !taskData->inputs.empty();
  }
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] * taskData->inputs_count[1] / world.size();
  }
  broadcast(world, delta, 0);
  local_input_.resize(delta);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + delta * proc, delta);
    }
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  int local_max = *std::max_element(local_input_.begin(), local_input_.end());
  if (world.rank() == 0) {
    res = local_max;
    for (int proc = 1; proc < world.size(); proc++) {
      int received_max;
      world.recv(proc, 0, received_max);
      if (received_max > res) {
        res = received_max;
      }
    }
  } else {
    world.send(0, 0, local_max);
  }
  return true;
}

bool opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}