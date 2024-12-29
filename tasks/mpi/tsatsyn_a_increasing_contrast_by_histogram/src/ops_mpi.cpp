// Copyright 2023 Nesterov Alexander
#include "mpi/tsatsyn_a_increasing_contrast_by_histogram/include/ops_mpi.hpp"

#include <algorithm>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->outputs_count[0] > 0 && taskData->inputs_count[0] > 0);
}
bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_data.resize(taskData->inputs_count[0]);
  auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tempPtr, tempPtr + taskData->inputs_count[0], input_data.begin());

  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  int min_val = *std::min_element(input_data.begin(), input_data.end());
  int max_val = *std::max_element(input_data.begin(), input_data.end());
  if (max_val - min_val == 0) {
    max_val++;
  }
  res.resize(input_data.size());
  int input_sz = static_cast<int>(input_data.size());
  for (int i = 0; i < input_sz; i++) {
    res[i] = (input_data[i] - min_val) * (255 - 0) / (max_val - min_val) + 0;
  }
  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* outputPtr = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res.begin(), res.end(), outputPtr);
  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->outputs_count[0] > 0 && taskData->inputs_count[0] > 0);
  }
  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_data.resize(taskData->inputs_count[0]);
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tempPtr, tempPtr + taskData->inputs_count[0], input_data.begin());
  }
  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  std::vector<int> local_data;
  int min_val;
  int max_val;
  int input_sz;
  input_sz = static_cast<int>(input_data.size());
  if (world.rank() == 0) {
    min_val = *std::min_element(input_data.begin(), input_data.end());
    max_val = *std::max_element(input_data.begin(), input_data.end());
    if (max_val - min_val == 0) {
      max_val++;
    }
    for (int proc = 1; proc < world.size(); proc++) {
      for (int i = proc; i < input_sz; i += world.size()) {
        local_data.emplace_back(input_data[i]);
      }
      world.send(proc, 0, local_data);
      local_data.clear();
    }
    for (int i = 0; i < input_sz; i += world.size()) {
      local_data.emplace_back(input_data[i]);
    }
  } else {
    world.recv(0, 0, local_data);
  }
  boost::mpi::broadcast(world, max_val, 0);
  boost::mpi::broadcast(world, min_val, 0);

  for (int i = 0; i < static_cast<int>(local_data.size()); i++) {
    local_data[i] = ((local_data[i] - min_val) * (255 - 0) / (max_val - min_val)) + 0;
  }
  if (world.rank() == 0) {
    std::vector<int> expected(1, 0);
    expected.resize(taskData->inputs_count[0]);
    for (int i = 0; i < static_cast<int>(local_data.size()); i++) {
      expected[i * world.size()] = local_data[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.recv(proc, 0, local_data);
      for (int i = 0; i < static_cast<int>(local_data.size()); i++) {
        expected[i * world.size() + proc] = local_data[i];
      }
    }
    res = expected;
  } else {
    world.send(0, 0, local_data);
  }

  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* outputPtr = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res.begin(), res.end(), outputPtr);
  }
  return true;
}