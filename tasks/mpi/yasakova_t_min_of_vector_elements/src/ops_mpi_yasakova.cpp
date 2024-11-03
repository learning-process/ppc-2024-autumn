// Copyright 2023 Nesterov Alexander
#include "mpi/yasakova_t_min_of_vector_elements/include/ops_mpi_yasakova.hpp"

#include <algorithm>
#include <random>
#include <vector>

bool yasakova_t_min_of_vector_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  inputValues_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], inputValues_[i].begin());
  }
  res_ = INT_MAX;
  return true;
}

bool yasakova_t_min_of_vector_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool yasakova_t_min_of_vector_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  std::vector<int> local_res(inputValues_.size());
  for (unsigned int i = 0; i < inputValues_.size(); i++) {
    local_res[i] = *std::min_element(inputValues_[i].begin(), inputValues_[i].end());
  }
  res_ = *std::min_element(local_res.begin(), local_res.end());
  return true;
}

bool yasakova_t_min_of_vector_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}

bool yasakova_t_min_of_vector_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  
  res_ = INT_MAX;
  return true;
}
bool yasakova_t_min_of_vector_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1 && !taskData->inputs.empty();
  }
  return true;
}
bool yasakova_t_min_of_vector_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] * taskData->inputs_count[1] / world.size();
  }
  broadcast(world, delta, 0);
  if (world.rank() == 0) {
    unsigned int rows = taskData->inputs_count[0];
    unsigned int columns = taskData->inputs_count[1];
    inputValues_ = std::vector<int>(rows * columns);
    for (unsigned int i = 0; i < rows; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      for (unsigned int j = 0; j < columns; j++) {
        inputValues_[i * columns + j] = tmp_ptr[j];
      }
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, inputValues_.data() + delta * proc, delta);
    }
  }
  localInputValues_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    localInputValues_ = std::vector<int>(inputValues_.begin(), inputValues_.begin() + delta);
  } else {
    world.recv(0, 0, localInputValues_.data(), delta);
  }
  int local_res = *std::min_element(localInputValues_.begin(), localInputValues_.end());
  reduce(world, local_res, res_, boost::mpi::minimum<int>(), 0);
  return true;
}
bool yasakova_t_min_of_vector_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}