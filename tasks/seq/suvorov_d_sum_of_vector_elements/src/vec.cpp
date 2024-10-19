// Copyright 2024 Nesterov Alexander
#include "seq/suvorov_d_sum_of_vector_elements/include/vec.hpp"

#include <thread>

using namespace std::chrono_literals;

bool suvorov_d_sum_of_vector_elements_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = reinterpret_cast<int*>(taskData->inputs[0])[0];
  res = 0;
  return true;
}

bool suvorov_d_sum_of_vector_elements_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1;
}

bool suvorov_d_sum_of_vector_elements_seq::TestTaskSequential::run() {
  internal_order_test();
  
  // ...

  return true;
}

bool suvorov_d_sum_of_vector_elements_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
