// Copyright 2024 Nesterov Alexander
#include "seq/zaitsev_a_min_of_vector_elements/include/ops_seq.hpp"

bool zaitsev_a_min_of_vector_elements_seq::MinOfVectorElementsSequential::pre_processing() {
  internal_order_test();

  // Init value for input and output
  input = std::vector<int>(taskData->inputs_count[0]);
  auto* interpreted_input = reinterpret_cast<int*>(taskData->inputs[0]);
  for (size_t i = 0; i < taskData->inputs_count[0]; i++) {
    input[i] = interpreted_input[i];
  }
  return true;
}

bool zaitsev_a_min_of_vector_elements_seq::MinOfVectorElementsSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs_count[0] != 0 && taskData->outputs_count[0] == 1) ||
         (taskData->inputs_count[0] == 0 && taskData->outputs_count[0] == 0);
}

bool zaitsev_a_min_of_vector_elements_seq::MinOfVectorElementsSequential::run() {
  internal_order_test();

  int currentMin = input[0];
  for (auto i : input) currentMin = (currentMin > i) ? i : currentMin;
  res = currentMin;
  return true;
}

bool zaitsev_a_min_of_vector_elements_seq::MinOfVectorElementsSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
