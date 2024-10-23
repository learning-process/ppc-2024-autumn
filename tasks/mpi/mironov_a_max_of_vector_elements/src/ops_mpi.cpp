// Copyright 2023 Nesterov Alexander
#include "mpi/mironov_a_max_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool mironov_a_max_of_vector_elements_mpi::MaxVectorSequential::pre_processing() {
  internal_order_test();

  // Init value for input and output
  input_ = std::vector<int32_t>(taskData->inputs_count[0]);
  int32_t* it = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  for (int32_t i = 0; i < input_.size(); ++i) {
    input_[i] = it[i];
  }
  res = input_[0];
  return true;
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs_count[0] > 0) && (taskData->outputs_count[0] == 1);
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorSequential::run() {
  internal_order_test();
  
  for (int32_t i = 1; i < input_.size(); ++i) {
      if (res < input_[i]) res = input_[i];
  }

  return true;
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int32_t*>(taskData->outputs[0])[0] = res;
  return true;
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorMPI::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  std::cout << "broadcast in: " << world.rank() << " 1 " << std::endl;
  broadcast(world, delta, 0);
  std::cout << "bradcast out: " << world.rank() << " 1 " << std::endl;

  if (world.rank() == 0) {
    // Init vectors
   
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_ = std::vector<int>(taskData->inputs_count[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      std::cout << "Send in: " << world.rank() << " 1 " << std::endl;
      world.send(proc, 0, input_.data() + proc * delta, delta);
      std::cout << "Send out: " << world.rank() << " 1 " << std::endl;
    }
  }
  local_input_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    std::cout << "recv in: " << world.rank() << " 1 " << std::endl;
    world.recv(0, 0, local_input_.data(), delta);
    std::cout << "recv out: " << world.rank() << " 1 " << std::endl;
  }
  // Init value for output
  res = input_[0];
  return true;
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return (taskData->inputs_count[0] > 0) && (taskData->outputs_count[0] == 1);
  }
  return true;
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorMPI::run() {
  internal_order_test();
  int local_res = *std::max_element(local_input_.begin(), local_input_.end());
  std::cout << "reduce in: " << world.rank() << " 1 " << std::endl;
  reduce(world, local_res, res, boost::mpi::maximum<int>(), 0);
  std::cout << "reduce out: " << world.rank() << " 1 " << std::endl;
  
  return true;
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int32_t*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
