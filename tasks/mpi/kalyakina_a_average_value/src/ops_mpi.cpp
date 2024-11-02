// Copyright 2023 Nesterov Alexander
#include "mpi/kalyakina_a_average_value/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

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
  for (unsigned int i = 0; i < input_vector.size(); i++) {
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
  if (world.rank() == 0) {
    // Init vectors
    input_vector = std::vector<int>(taskData->inputs_count[0]);
    int* it = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(it, it + taskData->inputs_count[0], input_vector.begin());
  }
  
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
  unsigned int part = 0;
  unsigned int reminder = 0;
  if (world.rank() == 0) {
    part = taskData->inputs_count[0] / world.size();
    reminder = taskData->inputs_count[0] % world.size();
  }
  boost::mpi::broadcast(world, part, 0);
  boost::mpi::broadcast(world, reminder, 0);
  std::vector<int> distr(world.size(), part);
  std::vector<int> displ(world.size());
  for (unsigned int i = 0; i < static_cast<unsigned int>(world.size()); i++) {
    if (reminder > 0) {
      distr[i]++;
      reminder--;
    }
    if (i == 0) {
      displ[i] = 0;
    } else {
      displ[i] = displ[i - 1] + distr[i - 1];
    }
  }
  local_input_vector = std::vector<int>(distr[world.rank()]);
  boost::mpi::scatterv(world, input_vector.data(), distr, displ, local_input_vector.data(), distr[world.rank()], 0);

  int local_res = 0;
  for (unsigned int i = 0; i < local_input_vector.size(); i++) {
    local_res += local_input_vector[i];
  }
  boost::mpi::reduce(world, local_res, result, std::plus<>(), 0);
  return true;
}

bool kalyakina_a_average_value_mpi::FindingAverageMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = (double)result / taskData->inputs_count[0];
  }
  return true;
}
