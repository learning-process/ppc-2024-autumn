// Copyright 2023 Nesterov Alexander
#include "mpi/tsatsyn_a_increasing_contrast_by_histogram/include/ops_mpi.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0);
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
  std::vector<double> local_data;
  int width, height;
  if (world.rank() == 0) {
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[1]);
    width = tempPtr[0];
    height = tempPtr[1];

    for (int proc = 1; proc < world.size(); proc++) {
      local_data.clear();
      for (int i = proc; i < static_cast<int>(input_data.size()); i += world.size()) {
        local_data.emplace_back(input_data[i]);
      }
      world.send(proc, 0, local_data);
    }
    local_data.clear();
    for (int i = 0; i < static_cast<int>(input_data.size()); i += world.size()) {
      local_data.emplace_back(input_data[i]);
    }
  } else {
    world.recv(0, 0, local_data);
  }
  boost::mpi::broadcast(world, height, 0);
  boost::mpi::broadcast(world, width, 0);
  std::vector<double> numbers(256, 0);
  for (int i = 0; i < static_cast<int>(local_data.size()); i++) {
    numbers[local_data[i]]++;
  }
  if (world.rank() == 0) {
    std::vector<double> received_numbers(256, 0);
    for (int proc = 1; proc < world.size(); proc++) {
      world.recv(proc, 0, received_numbers);
      for (int i = 0; i < static_cast<int>(numbers.size()); i++) {
        numbers[i] += received_numbers[i];
      }
    }
  } else {
    world.send(0, 0, numbers);
  }
  if (world.rank() == 0) {
    for (int i = 0; i < static_cast<int>(numbers.size()); i++) {
      numbers[i] /= width;
      std::cout << numbers[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << numbers[0] << " ";
    for (int i = 1; i < static_cast<int>(numbers.size()); i++) {
      numbers[i] = (numbers[i] + numbers[i - 1]);
      std::cout << numbers[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    for (int i = 0; i < static_cast<int>(numbers.size()); i++) {
      numbers[i] *= 255;
      std::cout << numbers[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    for (int i = 0; i < static_cast<int>(numbers.size()); i++) {
      numbers[i] = round(numbers[i]);
      std::cout << numbers[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
  }
  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}