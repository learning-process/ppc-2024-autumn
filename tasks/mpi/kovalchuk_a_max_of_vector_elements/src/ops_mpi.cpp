// Copyright 2023 Nesterov Alexander
#include "mpi/kovalchuk_a_max_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

std::vector<int> kovalchuk_a_max_of_vector_elements::getRandomVector(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

std::vector<std::vector<int>> kovalchuk_a_max_of_vector_elements::getRandomMatrix(int rows, int columns, int min,
                                                                                  int max) {
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = kovalchuk_a_max_of_vector_elements::getRandomVector(columns, min, max);
  }
  return vec;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], input_[i].begin());
  }
  // Init value for output
  res_ = INT_MIN;
  return true;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskSequential::run() {
  internal_order_test();
  std::vector<int> local_res(input_.size());
  for (unsigned int i = 0; i < input_.size(); i++) {
    local_res[i] = *std::max_element(input_[i].begin(), input_[i].end());
  }
  res_ = *std::max_element(local_res.begin(), local_res.end());
  return true;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] * taskData->inputs_count[1] / world.size();
  }
  broadcast(world, delta, 0);
  if (world.rank() == 0) {
    // Init vectors
    unsigned int rows = taskData->inputs_count[0];
    unsigned int columns = taskData->inputs_count[1];
    input_ = std::vector<int>(rows * columns);
    for (unsigned int i = 0; i < rows; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      for (unsigned int j = 0; j < columns; j++) {
        input_[i * columns + j] = tmp_ptr[j];
      }
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + delta * proc, delta);
    }
  }
  local_input_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  // Init value for output
  res_ = INT_MIN;
  return true;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1 && !taskData->inputs.empty();
  }
  return true;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskParallel::run() {
  internal_order_test();
  int local_res = *std::max_element(local_input_.begin(), local_input_.end());
  reduce(world, local_res, res_, boost::mpi::maximum<int>(), 0);
  return true;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}