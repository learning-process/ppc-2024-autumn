// Copyright 2024 Sedova Olga
#include "mpi/sedova_o_max_of_vector_elements/include/ops_mpi.hpp"

#include <mpi.h>

#include <random>

int find_max_of_matrix(std::vector<int> &matrix) {
  if (matrix.empty()) {
    return std::numeric_limits<int>::min();
  }
  auto max_it = std::max_element(matrix.begin(), matrix.end());
  return *max_it;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[i]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], input_[i].begin());
  }
  res_ = INT_MIN;
  return true;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 1 && taskData->inputs_count[1] >= 1 && taskData->outputs_count[0] == 1;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  std::vector<int> local_(input_.size());
  for (unsigned int i = 0; i < input_.size(); i++) {
    local_[i] = *std::max_element(input_[i].begin(), input_[i].end());
  }
  res_ = *std::max_element(local_.begin(), local_.end());
  return true;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = res_;
  return true;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    unsigned int rows = taskData->inputs_count[0];
    unsigned int cols = taskData->inputs_count[1];
    input_ = std::vector<int>(rows * cols);
    for (unsigned int i = 0; i < rows; i++) {
      auto *input_data = reinterpret_cast<int *>(taskData->inputs[i]);
      for (unsigned int j = 0; j < cols; j++) {
        input_[i * cols + j] = input_data[j];
      }
    }
  }
  return true;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1 && !taskData->inputs.empty();
  }
  return true;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int a = 0;
  if (world.rank() == 0) {
    a = taskData->inputs_count[0] * taskData->inputs_count[1] / world.size();
  }
  broadcast(world, a, 0);
  if (world.rank() == 0) {
    unsigned int rows = taskData->inputs_count[0];
    unsigned int cols = taskData->inputs_count[1];
    input_ = std::vector<int>(rows * cols);
    for (unsigned int i = 0; i < rows; i++) {
      auto *tmp_ = reinterpret_cast<int *>(taskData->inputs[i]);
      for (unsigned int j = 0; j < cols; j++) {
        input_[i * cols + j] = tmp_[j];
      }
    }
    for (int i = 1; i < world.size(); i++) {
      world.send(i, 0, input_.data() + a * i, a);
    }
  }
  loc_input_ = std::vector<int>(a);
  if (world.rank() == 0) {
    loc_input_ = std::vector<int>(input_.begin(), input_.begin() + a);
  } else {
    world.recv(0, 0, loc_input_.data(), a);
  }
  int loc_res = *std::max_element(loc_input_.begin(), loc_input_.end());
  reduce(world, loc_res, res_, boost::mpi::maximum<int>(), 0);
  return true;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = res_;
  }
  return true;
}
