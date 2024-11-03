// Copyright 2024 Khovansky Dmitry
#include "mpi/khovansky_d_max_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int VectorMax(std::vector<int, std::allocator<int>> r) {
  if (r.empty()) {
    return 0;
  }

  int max = r[0];
  for (size_t i = 1; i < r.size(); i++) {
    if (r[i] > max) {
      max = r[i];
    }
  }
  return max;
}

bool khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPISequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  return true;
}

bool khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPISequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPISequential::run() {
  internal_order_test();
  if (input_.empty()) {
    // Handle the case when the input vector is empty
    return true;
  }
  res = VectorMax(input_);
  return true;
}

bool khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPISequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPIParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPIParallel::validation() {
  internal_order_test();
  if (world_.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPIParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  if (world_.rank() == 0) {
    delta = taskData->inputs_count[0] / world_.size();
  }
  broadcast(world_, delta, 0);

  if (world_.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world_.size(); proc++) {
      world_.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }
  local_input_ = std::vector<int>(delta);
  if (world_.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world_.recv(0, 0, local_input_.data(), delta);
  }
  if (local_input_.empty()) {
    // Handle the case when the local input vector is empty
    return true;
  }
  int max = VectorMax(local_input_);

  reduce(world_, max, res_, boost::mpi::maximum<int>(), 0);
  return true;
}

bool khovansky_d_max_of_vector_elements_mpi::MaxOfVectorMPIParallel::post_processing() {
  internal_order_test();
  if (world_.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}
