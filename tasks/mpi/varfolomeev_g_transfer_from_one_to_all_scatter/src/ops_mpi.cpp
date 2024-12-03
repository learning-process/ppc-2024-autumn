// Copyright 2023 Nesterov Alexander
#include "mpi/varfolomeev_g_transfer_from_one_to_all_scatter/include/ops_mpi.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential::pre_processing() {
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

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  if (ops == "+") {
    res = std::accumulate(input_.begin(), input_.end(), 0);
  } else if (ops == "-") {
    res = -std::accumulate(input_.begin(), input_.end(), 0);
  } else if (ops == "max") {
    res = *std::max_element(input_.begin(), input_.end());
  }
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// bc - все данные  scatter - часть данных

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_values.resize(taskData->inputs_count[0]);
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tempPtr, tempPtr + taskData->inputs_count[0], input_values.begin());
  }
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.size() < 0 || world.rank() >= world.size()) {
    return false;
  }
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int proc_num = 1; proc_num < world.size(); proc_num++) {
      std::vector<int> local_input_data;
      for (int i = proc_num - 1; i < input_values.size(); i += (world.size() - 1)) {
        local_input_data.push_back(input_values[i]);
      }
      world.send(proc_num, 0, local_input_data);
    }
  } else {
    world.recv(0, 0, local_input_values);
    std::cout << local_input_values.size() << std::endl;
  }
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
