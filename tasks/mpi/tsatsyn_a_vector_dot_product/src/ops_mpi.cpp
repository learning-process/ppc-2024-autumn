// Copyright 2023 Nesterov Alexander
#include "mpi/tsatsyn_a_vector_dot_product/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

int tsatsyn_a_vector_dot_product_mpi::resulting(const std::vector<int>& v1, const std::vector<int>& v2) {
  int res = 0;
  for (size_t i = 0; i < v1.size(); ++i) {
    res += v1[i] * v2[i];
  }
  return res;
}
std::vector<int> tsatsyn_a_vector_dot_product_mpi::toGetRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
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
    delta = taskData->inputs_count[0] / world.size();
    // Init vectors
    v1 = std::vector<int>(taskData->inputs_count[0]);
    v2 = std::vector<int>(taskData->inputs_count[1]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    auto* tmp_ptr2 = reinterpret_cast<int*>(taskData->inputs[1]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      v1[i] = tmp_ptr[i];
      v2[i] = tmp_ptr2[i];
    }
    for (int proc = 1; proc < world.size(); proc++) world.send(proc, 0, delta);
  } else {
  }
  // Init value for output
  res = 0;
  return true;
}

bool tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return (taskData->inputs_count[0] == taskData->inputs_count[1]) &&
           (taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 2) &&
           taskData->outputs_count[0] == 1 && (taskData->outputs.size() == taskData->outputs_count.size()) &&
           taskData->outputs.size() == 1;
  }
  return true;
}

bool tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  // broadcast(world, delta, 0);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); ++proc) world.send(proc, 0, delta);
  }
  local_v1.resize(delta);
  local_v2.resize(delta);
  if (world.rank() == 0) {
    std::copy(v1.begin(), v1.begin() + delta, local_v1.begin());
    std::copy(v2.begin(), v2.begin() + delta, local_v2.begin());
  } else {
    world.recv(0, 0, delta);
  }
  int local_result = std::inner_product(local_v1.begin(), local_v1.end(), local_v2.begin(), 0);
  std::vector<int> full_results;
  gather(world, local_result, full_results, 0);

  if (world.rank() == 0) {
    res = std::accumulate(full_results.begin(), full_results.end(), 0);
  }
  if (world.rank() == 0 && (int)(taskData->inputs_count[0]) < world.size()) {
    res = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0);
  }
  return true;
}

bool tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
