// Copyright 2023 Nesterov Alexander
#include "mpi/example/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> suvorov_d_linear_topology::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool suvorov_d_linear_topology::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  std::uint32_t data_size = taskData->inputs_count[0];
  local_data_.resize(data_size);

  if (world.rank() == 0) {
    // Init data vector in proc with rank 0
    int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + data_size, local_data_.begin());
  }

  return true;
}

bool suvorov_d_linear_topology::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count.size() == 2 && taskData->outputs_count[0] > 0;
  }
  return true;
}

bool suvorov_d_linear_topology::TestMPITaskParallel::run() {
  internal_order_test();

  if (world.rank() == 0) {
    rank_order.push_back(0);
    send(1, 0, local_data_);
    send(1, 1, rank_order.size());
    send(1, 2, rank_order);
  } else {
    int order_size;
    recv(world.rank() - 1, 0, local_input_);
    recv(world.rank() - 1, 1, order_size);
    recv(world.rank() - 1, 2, rank_order);

    rank_order.push_back(world.rank());

    if (world.rank() != world.size() - 1) {
      send(world.rank() + 1, 0, local_input_);
      send(world.rank() + 1, 1, rank_order.size());
      send(world.rank() + 1, 2, rank_order);
    }
  }

  return true;
}

bool suvorov_d_linear_topology::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == world.size() - 1) {
    int* output_data_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(local_input_.begin(), local_input_.end(), output_data_ptr);
    taskData->outputs_count[0] = local_input_.size();
    
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(rank_order.data()));
    taskData->outputs_count.emplace_back(rank_order.size());
  }
  return true;
}
