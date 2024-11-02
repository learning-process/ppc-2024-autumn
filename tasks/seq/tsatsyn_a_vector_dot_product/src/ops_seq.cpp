// Copyright 2024 Nesterov Alexander
#include "seq/tsatsyn_a_vector_dot_product/include/ops_seq.hpp"

#include <random>
#include <thread>
using namespace std::chrono_literals;

int tsatsyn_a_vector_dot_product_seq::resulting(const std::vector<int>& v1, const std::vector<int>& v2) {
  int res = 0;
  for (size_t i = 0; i < v1.size(); ++i) {
    res += v1[i] * v2[i];
  }
  return res;
}

bool tsatsyn_a_vector_dot_product_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  v1.resize(taskData->inputs_count[0]);
  v2.resize(taskData->inputs_count[1]);
  auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tempPtr, tempPtr + taskData->inputs_count[0], v1.begin());
  tempPtr = reinterpret_cast<int*>(taskData->inputs[1]);
  std::copy(tempPtr, tempPtr + taskData->inputs_count[1], v2.begin());
  res = 0;
  return true;
}

bool tsatsyn_a_vector_dot_product_seq::TestTaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] == taskData->inputs_count[1]) &&
         (taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 2) &&
         taskData->outputs_count[0] == 1 && (taskData->outputs.size() == taskData->outputs_count.size()) &&
         taskData->outputs.size() == 1;
}

bool tsatsyn_a_vector_dot_product_seq::TestTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < v1.size(); i++) {
    res += v1[i] * v2[i];
  }
  return true;
}

bool tsatsyn_a_vector_dot_product_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
