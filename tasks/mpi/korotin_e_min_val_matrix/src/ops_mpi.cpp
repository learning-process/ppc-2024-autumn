// Copyright 2023 Nesterov Alexander
#include "mpi/korotin_e_min_val_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<double> korotin_e_min_val_matrix_mpi::getRandomMatrix(const unsigned rows, const unsigned columns, double scal) {
  if (rows==0 || columns==0) {
    throw std::invalid_argument("Can't creaate matrix with 0 rows or columns");
  }


  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> matrix(rows * columns);
  for (unsigned i = 0; i < rows * columns; i++) {
    matrix[i] = gen() / scal;
  }
  return matrix;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init matrixes
  input_ = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init value for output
  res = 0.0;
  return true;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = *std::min_element(input_.begin(), input_.end());
  return true;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init matixes
    input_ = std::vector<double>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
    //world.send(world.size(), 0, input_.data() + (world.size() - 1) * delta, taskData->inputs_count[0] - (world.size() - 1) * delta);
  }
  local_input_ = std::vector<double>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<double>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  // Init value for output
  res = {};
  return true;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  double local_res;

  local_res = *std::min_element(local_input_.begin(), local_input_.end());

  reduce(world, local_res, res, boost::mpi::minimum<double>(), 0);
  
  std::this_thread::sleep_for(20ms);
  return true;
}

bool korotin_e_min_val_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
