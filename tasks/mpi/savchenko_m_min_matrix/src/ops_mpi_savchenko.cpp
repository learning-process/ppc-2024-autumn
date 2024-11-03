// Copyright 2023 Nesterov Alexander
#include "mpi/savchenko_m_min_matrix/include/ops_mpi_savchenko.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

// Task Sequential

bool savchenko_m_min_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  rows = taskData->inputs_count[0];
  columns = taskData->inputs_count[1];
  matrix = std::vector<int>(rows * columns);

  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + rows * columns, matrix.begin());

  // Init value for output
  res = INT_MAX;
  return true;
}

bool savchenko_m_min_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool savchenko_m_min_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  res = *std::min_element(matrix.begin(), matrix.end());
  return true;
}

bool savchenko_m_min_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

// Task Parallel

bool savchenko_m_min_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] * taskData->inputs_count[1] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init vectors

    rows = taskData->inputs_count[0];
    columns = taskData->inputs_count[1];
    matrix = std::vector<int>(rows * columns);

    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + rows * columns, matrix.begin());

    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, matrix.data() + delta * proc, delta);
    }
  }

  local_matrix = std::vector<int>(delta);
  if (world.rank() == 0) {
    local_matrix = std::vector<int>(matrix.begin(), matrix.begin() + delta);
  } else {
    world.recv(0, 0, local_matrix.data(), delta);
  }

  // Init value for output
  res = INT_MAX;
  return true;
}

bool savchenko_m_min_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
  }
  return true;
}

bool savchenko_m_min_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  local_res = *std::min_element(local_matrix.begin(), local_matrix.end());
  reduce(world, local_res, res, boost::mpi::minimum<int>(), 0);

  return true;
}

bool savchenko_m_min_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}