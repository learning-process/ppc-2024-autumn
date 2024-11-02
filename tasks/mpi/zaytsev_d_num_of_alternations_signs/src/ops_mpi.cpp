// Copyright 2023 Nesterov Alexander
#include "mpi/zaytsev_d_num_of_alternations_signs/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> zaytsev_d_num_of_alternations_signs_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
    int input_count = taskData->inputs_count[0];
    data_.assign(input_data, input_data + input_count); 
    res = 0;
    return true;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  if (data_.size() < 2) {
        res = 0; 
        return true;
    }

    for (size_t i = 1; i < data_.size(); ++i) {
        if ((data_[i] >= 0 && data_[i - 1] < 0) || (data_[i] < 0 && data_[i - 1] >= 0)) {
            res++;
        }
    }

    return true; 
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel::pre_processing() {
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
  // Init value for output
  res = 0;
  return true;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int local_res;
  if (ops == "+") {
    local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  } else if (ops == "-") {
    local_res = -std::accumulate(local_input_.begin(), local_input_.end(), 0);
  } else if (ops == "max") {
    local_res = *std::max_element(local_input_.begin(), local_input_.end());
  }

  if (ops == "+" || ops == "-") {
    reduce(world, local_res, res, std::plus(), 0);
  } else if (ops == "max") {
    reduce(world, local_res, res, boost::mpi::maximum<int>(), 0);
  }
  return true;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
