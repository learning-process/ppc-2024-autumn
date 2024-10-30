// Copyright 2023 Nesterov Alexander
#include "mpi/kalyakina_a_average_value/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> kalyakina_a_average_value_mpi::RandomVectorWithFixSum(int sum, const int& count) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<int> result_vector;
  for (int i = 0; i < count - 1; i++) {
    result_vector.push_back(gen() % (__min(sum, 255) - 1));
    sum -= result_vector[i];
  }
  result_vector.push_back(sum);
  return result_vector;
}

bool kalyakina_a_average_value_mpi::FindingAverageMPITaskSequential::pre_processing() {
  internal_order_test();

  // Init value for input and output
  input_vector = std::vector<int>(taskData->inputs_count[0]);
  int* it = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(it, it + taskData->inputs_count[0], input_vector.begin());

  // Init value for output
  average_value = 0.0;
  return true;
}

bool kalyakina_a_average_value_mpi::FindingAverageMPITaskSequential::validation() {
  internal_order_test();

  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool kalyakina_a_average_value_mpi::FindingAverageMPITaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < input_vector.size(); i++) {
    average_value += input_vector[i];
  }
  average_value /= input_vector.size();
  return true;
}

bool kalyakina_a_average_value_mpi::FindingAverageMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = average_value;
  return true;
}

bool kalyakina_a_average_value_mpi::FindingAverageMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size() + 1;
  }
  broadcast(world, delta, 0);
  
  if (world.rank() == 0) {
    // Init vectors
    input_vector = std::vector<int>(taskData->inputs_count[0]);
    int* it = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(it, it + taskData->inputs_count[0], input_vector.begin());
    input_vector.resize(delta * world.size(), 0);
  }
  local_input_vector = std::vector<int>(delta);
  boost::mpi::scatter(world, &input_vector[0], &local_input_vector[0], delta, 0);

  // Init value for output
  if (world.rank() == 0) {
    result = 0;
  }
  return true;
}

bool kalyakina_a_average_value_mpi::FindingAverageMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool kalyakina_a_average_value_mpi::FindingAverageMPITaskParallel::run() {
  internal_order_test();
  int local_res = 0;
  for (unsigned int i = 0; i < local_input_vector.size(); i++) {
    local_res += local_input_vector[i];
  }
  reduce(world, local_res, result, std::plus(), 0);
  return true;
}

bool kalyakina_a_average_value_mpi::FindingAverageMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = (double)result / taskData->inputs_count[0];
  }
  return true;
}
