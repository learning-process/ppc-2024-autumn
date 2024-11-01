// Copyright 2024 Nesterov Alexander
#include "seq/yasakova_t_min_of_vector_elements/include/ops_seq_yasakova.hpp"

#include <climits>
#include <random>

using namespace std::chrono_literals;

std::vector<int> yasakova_t_min_of_vector_elements_seq::RandomVector(int size, int minimum, int maximum) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = minimum + gen() % (maximum - minimum + 1);
  }
  return vec;
}

std::vector<std::vector<int>> yasakova_t_min_of_vector_elements_seq::RandomMatrix(int rows, int columns, int minimum,
                                                                                  int maximum) {
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = yasakova_t_min_of_vector_elements_seq::RandomVector(columns, minimum, maximum);
  }
  return vec;
}

bool yasakova_t_min_of_vector_elements_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));

  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  res_ = INT_MAX;
  return true;
}

bool yasakova_t_min_of_vector_elements_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] == 1;
}

bool yasakova_t_min_of_vector_elements_seq::TestTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size(); i++) {
    for (size_t j = 0; j < input_[i].size(); j++) {
      if (input_[i][j] < res_) {
        res_ = input_[i][j];
      }
    }
  }
  return true;
}

bool yasakova_t_min_of_vector_elements_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}
