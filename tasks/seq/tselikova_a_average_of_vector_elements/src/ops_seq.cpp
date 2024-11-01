// Copyright 2024 Tselikova Arina
#include "seq/tselikova_a_average_of_vector_elements/include/ops_seq.hpp"

#include <thread>
#include <vector>
#include <algorithm>

using namespace std::chrono_literals;

bool tselikova_a_average_of_vector_elements::TestTaskSequential::pre_processing() {
  internal_order_test();
  int* tmp = reinterpret_cast<int*>(taskData->inputs[0]);
  input_ = std::vector<int>(taskData->inputs_count[0]);
  for (int i = 0; i < (int)taskData->inputs_count[0]; i++) {
    input_[i] = tmp[i];
  }
  res = 0;
  return true;
}

bool tselikova_a_average_of_vector_elements::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool tselikova_a_average_of_vector_elements::TestTaskSequential::run() {
  internal_order_test();
  int sum = 0;
  for (int i = 0; i < input_.size(); i++) {
    sum += input_[i];
  }
  res = static_cast<float>(sum) / input_.size();
  return true;
}

bool tselikova_a_average_of_vector_elements::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<float*>(taskData->outputs[0])[0] = res;
  return true;
}
