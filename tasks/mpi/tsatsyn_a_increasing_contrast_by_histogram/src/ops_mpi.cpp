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
  if (world.rank() == 0) {
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[1]);
    int width, height;
    width = tempPtr[0];
    height = tempPtr[1];
    std::cout << height << width << std::endl;
    for (int proc = 1; proc < world.size(); proc++) {
      local_data.clear();
      for (int i = proc; i < input_data.size(); i += world.size()) {
        local_data.emplace_back(input_data[i]);
      }
      world.send(proc, 0, local_data);
    }
    local_data.clear();
    for (int i = 0; i < input_data.size(); i += world.size()) {
      local_data.emplace_back(input_data[i]);
    }
    //std::vector<double> localka(256, 0);
    //for (int i = 0; i < input_data.size(); i++) {
    //  localka[input_data[i]]++;
    //}
    //for (int i = 0; i < localka.size(); i++) {
    //  std::cout << localka[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;
    //std::cout << std::endl;
    //std::cout << std::endl;
    //for (int i = 0; i < localka.size(); i++) {
    //  localka[i] /= width;
    //  std::cout << localka[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;
    //std::cout << std::endl;
    //std::cout << std::endl;
    //std::cout << localka[0] << " ";
    //for (int i = 1; i < localka.size(); i++) {
    //  localka[i] = (localka[i] + localka[i - 1]);
    //  std::cout << localka[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;
    //std::cout << std::endl;
    //std::cout << std::endl;
    //for (int i = 0; i < localka.size(); i++) {
    //  localka[i] *= 255;
    //  std::cout << localka[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;
    //std::cout << std::endl;
    //std::cout << std::endl;
    //for (int i = 0; i < localka.size(); i++) {
    //  localka[i] = round(localka[i]);
    //  std::cout << localka[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;
    //std::cout << std::endl;
    //std::cout << std::endl;
  } else {
    world.recv(0, 0, local_data);
    std::vector<double> localka(256, 0);
  }
  std::cout << local_data.size();
  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}