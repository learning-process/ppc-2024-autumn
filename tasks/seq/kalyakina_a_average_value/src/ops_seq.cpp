// Copyright 2024 Nesterov Alexander
#include "seq/kalyakina_a_average_value/include/ops_seq.hpp"

#include <stdlib.h>

#include <random>
#include <thread>

using namespace std::chrono_literals;

std::vector<int> kalyakina_a_average_value_seq::RandomVectorWithFixSum(int sum, const int& count) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<int> result_vector(count);
  for (int i = 0; i < count - 1; i++) {
    result_vector[i] = gen() % (std::min(sum, 255) - 1);
    sum -= result_vector[i];
  }
  result_vector.push_back(sum);
  return result_vector;
}

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
