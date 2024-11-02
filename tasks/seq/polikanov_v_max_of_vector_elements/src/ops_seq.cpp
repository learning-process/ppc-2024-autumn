// Copyright 2024 Nesterov Alexander
#include "seq/polikanov_v_max_of_vector_elements/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
std::vector<int> polikanov_v_max_of_vector_elements::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool polikanov_v_max_of_vector_elements::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  int count = static_cast<int>(taskData->inputs_count[0]);
  int* input = reinterpret_cast<int*>(taskData->inputs[0]);
  input_ = std::vector<int>(count);
  for (int i = 0; i < count; ++i) {
    input_[i] = input[i];
  }
  res = INT_MIN;
  return true;
}

bool polikanov_v_max_of_vector_elements::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 0 && taskData->outputs_count[0] == 0 || taskData->outputs_count[0] == 1;
}

bool polikanov_v_max_of_vector_elements::TestTaskSequential::run() {
  internal_order_test();
  int count = input_.size();
  for (int i = 0; i < count; i++) {
    res = std::max(res, input_[i]);
  }
  return true;
}

bool polikanov_v_max_of_vector_elements::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
