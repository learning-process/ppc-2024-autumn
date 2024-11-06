// Copyright 2024 Nesterov Alexander
#include "seq/kalyakina_a_average_value/include/ops_seq.hpp"

#include <cstdlib>
#include <random>
#include <thread>

using namespace std::chrono_literals;

bool kalyakina_a_average_value_seq::FindingAverageOfVectorElementsTaskSequential::pre_processing() {
  internal_order_test();

  // Init value for input and output
  input_vector = std::vector<int>(taskData->inputs_count[0]);
  int* it = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(it, it + taskData->inputs_count[0], input_vector.begin());
  average_value = 0.0;
  return true;
}

bool kalyakina_a_average_value_seq::FindingAverageOfVectorElementsTaskSequential::validation() {
  internal_order_test();

  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool kalyakina_a_average_value_seq::FindingAverageOfVectorElementsTaskSequential::run() {
  internal_order_test();
  for (unsigned int i = 0; i < input_vector.size(); i++) {
    average_value += input_vector[i];
  }
  average_value /= input_vector.size();
  return true;
}

bool kalyakina_a_average_value_seq::FindingAverageOfVectorElementsTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = average_value;
  return true;
}
