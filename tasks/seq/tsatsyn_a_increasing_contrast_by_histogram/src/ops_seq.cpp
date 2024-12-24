// Copyright 2024 Nesterov Alexander
#include "seq/tsatsyn_a_increasing_contrast_by_histogram/include/ops_seq.hpp"

#include <algorithm>
#include <thread>
#include <vector>
bool tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential::validation() {
  internal_order_test();
  return (taskData->outputs_count[0] > 0 && taskData->inputs_count[0] > 0);
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
bool tsatsyn_a_increasing_contrast_by_histogram_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  auto* outputPtr = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res.begin(), res.end(), outputPtr);

  return true;
}
