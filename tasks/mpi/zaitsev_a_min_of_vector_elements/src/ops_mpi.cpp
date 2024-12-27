// Copyright 2023 Nesterov Alexander
#include "mpi/zaitsev_a_min_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>

bool zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsSequential::pre_processing() {
  internal_order_test();

  // Init value for input and output
  input = std::vector<int>(taskData->inputs_count[0]);
  auto* interpreted_input = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    input[i] = interpreted_input[i];
  }
  return true;
}

bool zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs_count[0] != 0 && taskData->outputs_count[0] == 1) ||
         (taskData->inputs_count[0] == 0 && taskData->outputs_count[0] == 0);
}

bool zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsSequential::run() {
  internal_order_test();

  int currentMin = input[0];
  for (auto i : input) currentMin = (currentMin > i) ? i : currentMin;
  res = currentMin;
  return true;
}

bool zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  }
  return true;
}

bool zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsParallel::run() {
  internal_order_test();

  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + (proc - 1) * delta, delta);
    }
  }
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin() + (delta * (world.size() - 1)), input_.end());
  } else {
    local_input_ = std::vector<int>(delta);
    world.recv(0, 0, local_input_.data(), delta);
  }

  int local_res = INT_MAX;
  if (!local_input_.empty()) {
    local_res = local_input_[0];
    for (auto i : local_input_) local_res = (local_res > i) ? i : local_res;
  }
  reduce(world, local_res, res, boost::mpi::minimum<int>(), 0);

  return true;
}

bool zaitsev_a_min_of_vector_elements_mpi::MinOfVectorElementsParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
