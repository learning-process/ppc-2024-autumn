// Copyright 2024 Nesterov Alexander
#include "seq/petrov_a_nearest_neighbor_elements/include/ops_seq.hpp"

#include <cmath>
#include <iostream>
#include <limits>

using namespace std::chrono_literals;

bool petrov_a_nearest_neighbor_elements_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  int size = taskData->inputs_count[0];
  input_.resize(size);

  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  for (int i = 0; i < size; ++i) {
    input_[i] = input_data[i];
  }


  res.resize(2); 
  return true;
}

bool petrov_a_nearest_neighbor_elements_seq::TestTaskSequential::validation() {
  internal_order_test();

  bool isValid = (!taskData->inputs_count.empty()) && (!taskData->inputs.empty()) && (!taskData->outputs.empty());

  return isValid;
}

bool petrov_a_nearest_neighbor_elements_seq::TestTaskSequential::run() {
  internal_order_test();

  size_t size = input_.size();
  if (size < 2) {
    return false;
  }
  int min_difference = std::numeric_limits<int>::max();
  size_t min_index = 0;

  for (size_t i = 0; i < size - 1; ++i) {
    int difference = std::abs(input_[i] - input_[i + 1]);
    if (difference < min_difference) {
      min_difference = difference;
      min_index = i;
    }
  }

  res[0] = input_[min_index];
  res[1] = input_[min_index + 1];

  return true;
}

bool petrov_a_nearest_neighbor_elements_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  int* output_ = reinterpret_cast<int*>(taskData->outputs[0]);
  output_[0] = res[0];
  output_[1] = res[1];

  return true;
}
