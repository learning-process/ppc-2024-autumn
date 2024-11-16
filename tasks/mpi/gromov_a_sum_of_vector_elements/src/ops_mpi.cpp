#include "mpi/gromov_a_sum_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  // Init value for output
  res = 0;
  return true;
}

bool gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential::run() {
  internal_order_test();
  if (ops == "add") {
    res = std::accumulate(input_.begin(), input_.end(), 0);
  } else if (ops == "max") {
    res = *std::max_element(input_.begin(), input_.end());
  } else if (ops == "min") {
    res = *std::min_element(input_.begin(), input_.end());
  }
  return true;
}

bool gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel::pre_processing() {
  internal_order_test();
  res = 0;
  return true;
}

bool gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  unsigned int alpha = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    alpha = taskData->inputs_count[0] % world.size();
  }
  broadcast(world, delta, 0);
  broadcast(world, alpha, 0);

  if (world.rank() == 0) {
    input_.assign(reinterpret_cast<int*>(taskData->inputs[0]),
                  reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0]);
    for (int proc = 1; proc < world.size(); ++proc) {
      unsigned int send_size = (proc == world.size() - 1) ? delta + alpha : delta;
      world.send(proc, 0, input_.data() + proc * delta, send_size);
    }
  }

  unsigned int local_size = (world.rank() == world.size() - 1) ? delta + alpha : delta;
  local_input_.resize(local_size);

  if (world.rank() != 0) {
    world.recv(0, 0, local_input_.data(), local_size);
  } else {
    std::copy(input_.begin(), input_.begin() + delta, local_input_.begin());
  }

  int local_res = 0;
  if (ops == "add") {
    local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  } else if (ops == "max") {
    local_res = *std::max_element(local_input_.begin(), local_input_.end());
  } else if (ops == "min") {
    local_res = *std::min_element(local_input_.begin(), local_input_.end());
  }

  if (ops == "add") {
    reduce(world, local_res, res, std::plus(), 0);
  } else if (ops == "max") {
    reduce(world, local_res, res, boost::mpi::maximum<int>(), 0);
  } else if (ops == "min") {
    reduce(world, local_res, res, boost::mpi::minimum<int>(), 0);
    if (world.rank() == 0) {
      if (input_.back() < res) {
        res = input_.back();
      }
    }
  }
  return true;
}

bool gromov_a_sum_of_vector_elements_mpi::MPISumOfVectorParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
