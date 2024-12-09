// Copyright 2024 Nesterov Alexander
#include "seq/tsatsyn_a_increasing_contrast_by_histogram/include/ops_seq.hpp"

#include <cmath>
#include <thread>

bool tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential::validation() {
  internal_order_test();
  return (taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0);

  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_data.resize(taskData->inputs_count[0]);
  auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tempPtr, tempPtr + taskData->inputs_count[0], input_data.begin());

  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential::run() {
  internal_order_test();
  auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[1]);
  int width, height;
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

  for (int i = 0; i < localka.size(); i++) {
    localka[i] = std::round(localka[i]);
    std::cout << localka[i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  return true;
}
bool tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
