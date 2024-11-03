// Copyright 2023 Nasedkin Egor
#include "mpi/nasedkin_e_matrix_column_max_value/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> nasedkin_e_matrix_column_max_value_mpi::getRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init value for output
  res_ = std::vector<int>(taskData->outputs_count[0], 0);
  return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == taskData->inputs_count[0] / taskData->inputs_count[1];
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxSequential::run() {
  internal_order_test();
  int rows = taskData->inputs_count[0] / taskData->inputs_count[1];
  int cols = taskData->inputs_count[1];
  for (int col = 0; col < cols; col++) {
    int max_val = input_[col];
    for (int row = 1; row < rows; row++) {
      max_val = std::max(max_val, input_[row * cols + col]);
    }
    res_[col] = max_val;
  }
  return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxSequential::post_processing() {
  internal_order_test();
  for (unsigned i = 0; i < res_.size(); i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
  }
  return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxParallel::pre_processing() {
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
  res_ = std::vector<int>(taskData->outputs_count[0], 0);
  return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == taskData->inputs_count[0] / taskData->inputs_count[1];
  }
  return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxParallel::run() {
  internal_order_test();
  int rows = taskData->inputs_count[0] / taskData->inputs_count[1];
  int cols = taskData->inputs_count[1];
  std::vector<int> local_res(cols, 0);
  for (int col = 0; col < cols; col++) {
    int max_val = local_input_[col];
    for (int row = 1; row < rows; row++) {
      max_val = std::max(max_val, local_input_[row * cols + col]);
    }
    local_res[col] = max_val;
  }

  // Используем векторы вместо указателей на данные
  reduce(world, local_res, res_, boost::mpi::maximum<int>(), 0);
  return true;
}

bool nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (unsigned i = 0; i < res_.size(); i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
    }
  }
  return true;
}