// Copyright 2023 Nesterov Alexander
#include "mpi/savchenko_m_min_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> savchenko_m_min_matrix_mpi::getRandomMatrix(int rows, int columns, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());

  // Forming a random matrix
  std::vector<int> matrix(rows * columns);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      matrix[i * columns + j] = min + gen() % (max - min + 1);
    }
  }

  return matrix;
}

// Task Sequential

bool savchenko_m_min_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool savchenko_m_min_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  columns = taskData->inputs_count[0];
  rows = taskData->inputs_count[1];
  matrix = std::vector<int>(rows * columns);

  for (int i = 0; i < rows; ++i) {
    auto *temp = reinterpret_cast<int *>(taskData->inputs[i]);

    for (int j = 0; j < columns; ++j) {
      matrix[i * columns + j] = temp[j];
    }
  }
  res = matrix[0];

  return true;
}

bool savchenko_m_min_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (int i = 0; i < matrix.size(); i++) {
    if (matrix[i] < res) {
      res = matrix[i];
    }
  }

  return true;
}

bool savchenko_m_min_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  return true;
}

// Task Parallel

bool savchenko_m_min_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1 && !taskData->inputs.empty();
  }
  return true;
}

bool savchenko_m_min_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] * taskData->inputs_count[1] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init vectors

    // WIP HERE

    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
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

bool savchenko_m_min_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int local_res;
  if (ops == "+") {
    local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  } else if (ops == "-") {
    local_res = -std::accumulate(local_input_.begin(), local_input_.end(), 0);
  } else if (ops == "max") {
    local_res = *std::max_element(local_input_.begin(), local_input_.end());
  }

  if (ops == "+" || ops == "-") {
    reduce(world, local_res, res, std::plus(), 0);
  } else if (ops == "max") {
    reduce(world, local_res, res, boost::mpi::maximum<int>(), 0);
  }
  return true;
}

bool savchenko_m_min_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = res;
  }
  return true;
}
