// Copyright 2024 Nesterov Alexander
#include "seq/seq_titov_s_vector_sum/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool titov_s_vector_sum_seq::VectorSumSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = reinterpret_cast<int*>(taskData->inputs[0])[0];
  res = 0;
  return true;
}

bool titov_s_vector_sum_seq::VectorSumSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool titov_s_vector_sum_seq::VectorSumSequential::run() {
  internal_order_test();
  res = 0;

  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int num_elements = taskData->inputs_count[0];

  for (int i = 0; i < num_elements; ++i) {
    res += input_data[i];
  }
  return true;
}

bool titov_s_vector_sum_seq::VectorSumSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
