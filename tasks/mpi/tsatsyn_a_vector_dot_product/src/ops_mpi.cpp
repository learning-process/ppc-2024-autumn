// Copyright 2023 Nesterov Alexander
#include "mpi/tsatsyn_a_vector_dot_product/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int tsatsyn_a_vector_dot_product_mpi::resulting(const std::vector<int>& v1, const std::vector<int>& v2) {
  int64_t res = 0;
  for (size_t i = 0; i < v1.size(); ++i) {
    res += v1[i] * v2[i];
  }
  return res;
}

bool tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  v1.resize(taskData->inputs_count[0]);
  v2.resize(taskData->inputs_count[1]);
  auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tempPtr, tempPtr + taskData->inputs_count[0], v1.begin());
  tempPtr = reinterpret_cast<int*>(taskData->inputs[1]);
  std::copy(tempPtr, tempPtr + taskData->inputs_count[0], v2.begin());
  res = 0;
  return true;
}

bool tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs_count[0] == taskData->inputs_count[1]) &&
         (taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 2) &&
         taskData->outputs_count[0] == 1 && (taskData->outputs.size() == taskData->outputs_count.size()) &&
         taskData->outputs.size() == 1;
}

bool tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < v1.size(); i++) res += v1[i] * v2[i];
  return true;
}

bool tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    (int)(taskData->inputs_count[0]) < world.size() ? delta = taskData->inputs_count[0]
                                                    : delta = taskData->inputs_count[0] / world.size();
    for (size_t i = 0; i < taskData->inputs.size(); ++i) {
      if (taskData->inputs[i] == nullptr || taskData->inputs_count[i] == 0) {
        return false;
      }
    }
    v1.resize(taskData->inputs_count[0]);
    int* source_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(source_ptr, source_ptr + taskData->inputs_count[0], v1.begin());

    v2.resize(taskData->inputs_count[1]);
    source_ptr = reinterpret_cast<int*>(taskData->inputs[1]);
    std::copy(source_ptr, source_ptr + taskData->inputs_count[1], v2.begin());
  }
  return true;
}

bool tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty() ||
        taskData->inputs_count[0] != taskData->inputs_count[1] || taskData->outputs_count[0] == 0) {
      return false;
    }
  }
  return true;
}

bool tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  broadcast(world, delta, 0);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); ++proc) {
      world.send(proc, 0, v1.data() + proc * delta, delta);
      world.send(proc, 1, v2.data() + proc * delta, delta);
    }
  }
  local_v1.resize(delta);
  local_v2.resize(delta);
  if (world.rank() == 0) {
    std::copy(v1.begin(), v1.begin() + delta, local_v1.begin());
    std::copy(v2.begin(), v2.begin() + delta, local_v2.begin());
  } else {
    world.recv(0, 0, local_v1.data(), delta);
    world.recv(0, 1, local_v2.data(), delta);
  }
  int local_result = 0;
  for (size_t i = 0; i < local_v1.size(); ++i) {
    local_result += local_v1[i] * local_v2[i];
  }
  std::vector<int> full_results;
  gather(world, local_result, full_results, 0);
  res = 0;
  if (world.rank() == 0) {
    for (int result : full_results) {
      res += result;
    }
  }
  if (world.rank() == 0 && (int)(taskData->inputs_count[0]) < world.size()) {
    res = 0;
    for (size_t i = 0; i < v1.size(); ++i) {
      res += v1[i] * v2[i];
    }
  }
  return true;
}

bool tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    if (!taskData->outputs.empty()) {
      reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
    } else {
      return false;
    }
  }
  return true;
}