// Copyright 2023 Nesterov Alexander
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "seq/kovalchuk_a_max_of_vector_elements/include/ops_seq.hpp"

std::vector<int> kovalchuk_a_max_of_vector_elements_seq::getRandomVector(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

std::vector<std::vector<int>> kovalchuk_a_max_of_vector_elements_seq::getRandomMatrix(int rows, int columns, int min,
                                                                                      int max) {
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = kovalchuk_a_max_of_vector_elements_seq::getRandomVector(columns, min, max);
  }
  return vec;
}

bool kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], input_[i].begin());
  }
  // Init value for output
  res_ = INT_MIN;
  return true;
}

bool kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask::run() {
  internal_order_test();
  std::vector<int> local_res(input_.size());
  for (unsigned int i = 0; i < input_.size(); i++) {
    local_res[i] = *std::max_element(input_[i].begin(), input_[i].end());
  }
  res_ = *std::max_element(local_res.begin(), local_res.end());
  return true;
}

bool kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}