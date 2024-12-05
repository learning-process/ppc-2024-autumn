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

std::vector<int> varfolomeev_g_transfer_from_one_to_all_scatter_mpi::getRandomVector(int sz, int a, int b) {  // [a, b]
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % (b - a + 1) + a;
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

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs[0] == nullptr || taskData->outputs[0] == nullptr) {
      return false;
    }
    // input_values.resize(taskData->inputs_count[0]);
    // auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    // std::copy(tempPtr, tempPtr + taskData->inputs_count[0], input_values.begin());
    input_values.assign(reinterpret_cast<int*>(taskData->inputs[0]), reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0]);
  }
  res = 0;
  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.size() < 0 || world.rank() >= world.size() || (ops != "+" && ops != "-" && ops != "max")) {
    return false;
  }

  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int rank = world.rank();
  int world_size = world.size();

  // Spread the data
  if (rank == 0) {
    for (int proc_num = 1; proc_num < world_size; proc_num++) {
      std::vector<int> local_data;
      for (int i = proc_num - 1; i < (int)input_values.size(); i += (world.size() - 1)) {
        local_data.push_back(input_values[i]);
      }
      world.send(proc_num, 0, local_data);
    }
  } else {
    world.recv(0, 0, local_input_values);
  }

  // Выполнение операции
  int local_result = 0;
  if (ops == "+") {
    local_result = std::accumulate(local_input_values.begin(), local_input_values.end(), 0);
  } else if (ops == "-") {
    local_result = -std::accumulate(local_input_values.begin(), local_input_values.end(), 0);
  } else if (ops == "max") {
    if (!local_input_values.empty()) {
      local_result = local_input_values[0];
      for (int value : local_input_values) {
        if (value > local_result) {
          local_result = value;
        }
      }
    }
  }

  // Gathering results (root)
  if (rank == 0) {
    std::vector<int> results(world_size - 1);
    for (int proc_num = 1; proc_num < world_size; proc_num++) {
      world.recv(proc_num, 0, results[proc_num - 1]);
    }
    if (ops == "max") {
      res = *std::max_element(results.begin(), results.end());
    } else {
      res = std::accumulate(results.begin(), results.end(), 0);
    }
  } else {  // Sending results (non-root)
    world.send(0, 0, local_result);
  }

  return true;
}

bool varfolomeev_g_transfer_from_one_to_all_scatter_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  input_values.clear();
  local_input_values.clear();
  return true;
}
