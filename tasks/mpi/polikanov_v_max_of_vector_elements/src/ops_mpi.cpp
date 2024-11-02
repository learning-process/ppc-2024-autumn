// Copyright 2023 Nesterov Alexander
#include "mpi/polikanov_v_max_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> polikanov_v_max_of_vector_elements::getRandomVector(int sz, int lower, int upper) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = lower + gen() % (upper - lower + 1);
  }
  vec[sz - 1] = upper;
  return vec;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  int count = static_cast<int>(taskData->inputs_count[0]);
  int* input = reinterpret_cast<int*>(taskData->inputs[0]);
  input_ = std::vector<int>(count);
  for (int i = 0; i < count; ++i) {
    input_[i] = input[i];
  }
  res = INT_MIN;
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 0 && taskData->outputs_count[0] == 0 || taskData->outputs_count[0] == 1;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskSequential::run() {
  internal_order_test();
  int count = input_.size();
  for (int i = 0; i < count; i++) {
    res = std::max(res, input_[i]);
  }
  std::this_thread::sleep_for(20ms);
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskParallel::pre_processing() {
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
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskParallel::run() {
  internal_order_test();
  if (local_input_.empty()) {
    // Handle the case when the local input vector is empty
    return true;
  }
  int max = INT_MIN;
  for (size_t i = 0; i < local_input_.size(); ++i) {
    max = std::max(max, local_input_[i]);
  }

  reduce(world, max, res, boost::mpi::maximum<int>(), 0);
  return true;
}

bool polikanov_v_max_of_vector_elements::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
