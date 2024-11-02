// Copyright 2023 Nesterov Alexander
#include "mpi/suvorov_d_sum_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> suvorov_d_sum_of_vector_elements_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }

  return vec;
}

bool suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  // Init value for output
  res = 0;
  return true;
}

bool suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq::run() {
  internal_order_test();
  res = std::accumulate(input_.begin(), input_.end(), 0);
  return true;
}

bool suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel::pre_processing() {
  internal_order_test();
  int input_size = taskData->inputs_count[0];
  if (input_size == 0) {
    res = 0;
    return true;
  }

  if (input_size <= world.size()) {
    local_input_ = std::vector<int>(1, (world.rank() < input_size) ? input_[world.rank()] : 0);
  } else {
    unsigned int delta = input_size / world.size();
    int rest = input_size % world.size();
    unsigned int local_size = delta + (world.rank() < rest ? 1 : 0);
    local_input_ = std::vector<int>(local_size);

    if (world.rank() == 0) {
      // Init vectors
      input_ = std::vector<int>(input_size);
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
      std::copy(tmp_ptr, tmp_ptr + input_size, input_.begin());

      int beginning = 0;
      for (int proc = 1; proc < world.size(); ++proc) {
        int send_elems_count = delta + (proc < rest ? 1 : 0);
        world.send(proc, 0, input_.data() + beginning, send_elems_count);
        beginning += send_elems_count;
      }

      local_input_.assign(input_.begin(), input_.begin() + local_size);
      // Init value for output
    } else {
      world.recv(0, 0, local_input_.data(), local_size);
    }
  }

  res = 0;
  return true;
}

bool suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel::run() {
  internal_order_test();
  int local_res;

  local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);

  reduce(world, local_res, res, std::plus(), 0);

  return true;
}

bool suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
