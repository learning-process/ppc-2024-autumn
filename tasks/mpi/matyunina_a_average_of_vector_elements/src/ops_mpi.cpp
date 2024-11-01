// Copyright 2023 Nesterov Alexander
#include "mpi/matyunina_a_average_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool matyunina_a_average_of_vector_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  res_ = 0;
  return true;
}

bool matyunina_a_average_of_vector_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool matyunina_a_average_of_vector_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res_ = std::accumulate(input_.begin(), input_.end(), 0);
  res_ /= static_cast<int>(input_.size());
  return true;
}

bool matyunina_a_average_of_vector_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}

bool matyunina_a_average_of_vector_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  unsigned int remainder = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    remainder = taskData->inputs_count[0] % world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);

    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + delta * proc + remainder, delta);
    }
  }
  local_input_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta + remainder);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  res_ = 0;
  return true;
}

bool matyunina_a_average_of_vector_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool matyunina_a_average_of_vector_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int local_res = 0;
  for (size_t i = 0; i < local_input_.size(); i++) {
    local_res += local_input_[i];
  }

  std::vector<int> all;
  boost::mpi::gather(world, local_res, all, 0);

  if (world.rank() == 0) {
    for (int res : all) {
      res_ += res;
    }
    res_ /= static_cast<int>(input_.size());
  }
  return true;
}

bool matyunina_a_average_of_vector_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}
