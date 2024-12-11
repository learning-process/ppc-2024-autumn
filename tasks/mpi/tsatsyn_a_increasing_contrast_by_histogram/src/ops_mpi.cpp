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

bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0);

  return true;
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
  auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[1]);
  int width;
  int height;
  width = tempPtr[0];
  height = tempPtr[1];
  std::cout << height << width << std::endl;
  std::vector<double> localka(256, 0);
  for (int i = 0; i < static_cast<int>(input_data.size()); i++) {
    localka[input_data[i]]++;
  }
  for (int i = 0; i < static_cast<int>(localka.size()); i++) {
    std::cout << localka[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  for (int i = 0; i < static_cast<int>(localka.size()); i++) {
    localka[i] /= width;
    std::cout << localka[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << localka[0] << " ";
  for (int i = 1; i < static_cast<int>(localka.size()); i++) {
    localka[i] = (localka[i] + localka[i - 1]);
    std::cout << localka[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  for (int i = 0; i < static_cast<int>(localka.size()); i++) {
    localka[i] *= 255;
    std::cout << localka[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  for (int i = 0; i < static_cast<int>(localka.size()); i++) {
    localka[i] = std::round(localka[i]);
    std::cout << localka[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
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
    for (int i = 0; i < input_data.size(); i++) {
      std::cout << input_data[i] << " ";
    }
    std::cout << std::endl;
  }

  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  std::vector<int> local_data;
  int width;
  int height;
  int min_val;
  int max_val;
  int input_sz;
  input_sz = static_cast<int>(input_data.size());
  if (world.rank() == 0) {
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[1]);
    width = tempPtr[0];
    height = tempPtr[1];
    min_val = *std::min_element(input_data.begin(), input_data.end());
    max_val = *std::max_element(input_data.begin(), input_data.end());
    std::cout << max_val << " " << min_val << std::endl;
    for (int proc = 1; proc < world.size(); proc++) {
      local_data.clear();
      for (int i = proc; i < input_sz; i += world.size()) {
        local_data.emplace_back(input_data[i]);
      }
      world.send(proc, 0, local_data);
    }
    local_data.clear();
    for (int i = 0; i < input_sz; i += world.size()) {
      local_data.emplace_back(input_data[i]);
    }
  } else {
    world.recv(0, 0, local_data);
  }
  boost::mpi::broadcast(world, max_val, 0);
  boost::mpi::broadcast(world, min_val, 0);
  std::cout << world.rank() << " " << local_data.size() << std::endl;

  int lz = static_cast<int>(local_data.size());
  for (int i = 0; i < lz; i++) {
    local_data[i] = (local_data[i] - min_val) * (255 - 0) / (max_val - min_val) + 0;
    std::cout << local_data[i] << " ";
  }
  if (world.rank() == 0) {
    std::vector<double> expected;
    expected.resize(input_data.size());
    for (int i = 0; i < lz; i++) {
      int j = i + world.size();
      expected[j] = input_data[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, local_data);
      for (int i = 0; i < lz; i++) {
        int j = i + world.size() + proc;
        expected[j] = input_data[i];
      }
    }

  } else {
  }
  /*std::vector<double> numbers(256, 0);
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
  }*/
  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}