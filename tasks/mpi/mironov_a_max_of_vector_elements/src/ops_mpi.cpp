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
  input_ = std::vector<int>(taskData->inputs_count[0]);
  int* it = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(it, it + taskData->inputs_count[0], input_.begin());
  result_ = input_[0];
  return true;
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs_count[0] > 0) && (taskData->outputs_count[0] == 1);
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorSequential::run() {
  internal_order_test();

  result_ = input_[0];
  for (size_t it = 1; it < input_.size(); ++it) {
    if (result_ < input_[it]) result_ = input_[it];
  }
  return true;
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result_;
  return true;
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorMPI::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    if (taskData->inputs_count[0] % world.size() != 0u) delta++;
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init vectors
    int* it = reinterpret_cast<int*>(taskData->inputs[0]);
    input_ = std::vector<int>(static_cast<int>(delta) * world.size(), INT_MIN);
    std::copy(it, it + taskData->inputs_count[0], input_.begin());

    // Send data
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }

  local_input_ = std::vector<int>(delta, INT_MIN);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  // Init value for output
  result_ = local_input_[0];
  return true;
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output & input
    return (taskData->inputs_count[0] > 0) && (taskData->outputs_count[0] == 1);
  }
  return true;
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorMPI::run() {
  internal_order_test();
  int local_res = local_input_[0];
  for (size_t it = 1; it < local_input_.size(); ++it) {
    if (local_res < local_input_[it]) local_res = local_input_[it];
  }
  reduce(world, local_res, result_, boost::mpi::maximum<int>(), 0);
  return true;
}

bool mironov_a_max_of_vector_elements_mpi::MaxVectorMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = result_;
  }
  return true;
}
