// Copyright 2024 Nesterov Alexander
#include "seq/suvorov_d_sum_of_vector_elements/include/vec.hpp"

bool suvorov_d_sum_of_vector_elements_seq::Sum_of_vector_elements_seq::pre_processing() {
  internal_order_test();
  // Init value for input and output
  int* input_pointer = reinterpret_cast<int*>(taskData->inputs[0]);
  input_.assign(input_pointer, input_pointer + taskData->inputs_count[0]);
  return true;
}

bool suvorov_d_sum_of_vector_elements_seq::Sum_of_vector_elements_seq::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] >= 0 && taskData->outputs_count[0] == 1;
}

bool suvorov_d_sum_of_vector_elements_seq::Sum_of_vector_elements_seq::run() {
  internal_order_test();

  res_ = std::accumulate(input_.begin(), input_.end(), 0);

  return true;
}

bool suvorov_d_sum_of_vector_elements_seq::Sum_of_vector_elements_seq::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}
