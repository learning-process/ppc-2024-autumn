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
  std::uniform_int_distribution<int> dist(-1000, 1000);

  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }

  return vec;
}

bool suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel::pre_processing() {
  internal_order_test();
  int input_size;

  if (world_.rank() == 0) {
    input_size = taskData->inputs_count[0];
    input_ = std::vector<int>(input_size);
    auto *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + input_size, input_.begin());
  }
  broadcast(world_, input_size, 0);

  int rest = input_size % world_.size();
  std::vector<int> sizes(world_.size(), input_size / world_.size());
  std::vector<int> displacements(world_.size(), 0);
  int local_size;

  if (world_.rank() == 0) {
    for (int i = 0; i < rest; ++i) {
      sizes[i]++;
    }
    for (int i = 1; i < world_.size(); ++i) {
      displacements[i] = displacements[i - 1] + sizes[i - 1];
    }

    local_size = sizes[world_.rank()];
    local_input_.resize(local_size);

    scatterv(world_, input_, sizes, displacements, local_input_.data(), local_size, 0);
  } else {
    if (world_.rank() < rest) {
      sizes[world_.rank()]++;
    }
    local_size = sizes[world_.rank()];
    local_input_.resize(local_size);

    scatterv(world_, local_input_.data(), local_size, 0);
  }

  return true;
}

bool suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel::validation() {
  internal_order_test();
  if (world_.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1 && taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 0;
  }
  return true;
}

bool suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel::run() {
  internal_order_test();
  int local_res;

  local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);

  reduce(world_, local_res, res_, std::plus(), 0);

  return true;
}

bool suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel::post_processing() {
  internal_order_test();
  if (world_.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}
