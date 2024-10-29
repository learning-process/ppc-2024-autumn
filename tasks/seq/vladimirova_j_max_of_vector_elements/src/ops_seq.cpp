// Copyright 2024 Nesterov Alexander
#pragma once
#include "seq/vladimirova_j_max_of_vector_elements/include/ops_seq.hpp"

#include <random>
#include <thread>

using namespace std::chrono_literals;

std::vector<int> vladimirova_j_max_of_vector_elements_seq::CreateVector(size_t size, size_t spread_of_val) {
  if (size == 0) {
    throw "null size";
  }
  std::random_device dev;
  std::mt19937 random(dev());
  std::vector<int> v(size);
  for (size_t i = 0; i < size; i++) {
    v[i] = (random() % (2 * spread_of_val + 1)) - spread_of_val;
  }
  return v;
}

std::vector<std::vector<int>> vladimirova_j_max_of_vector_elements_seq::CreateInputMatrix(size_t row_c, size_t column_c,
                                                                                          size_t spread_of_val) {
  if ((row_c == 0) || (column_c == 0)) {
    throw "null size";
  }

  //  Init value for input and output
  std::vector<std::vector<int>> m(row_c);
  for (size_t i = 0; i < row_c; i++) {
    m[i] = vladimirova_j_max_of_vector_elements_seq::CreateVector(column_c, spread_of_val);
  }
  return m;
}

int vladimirova_j_max_of_vector_elements_seq::FindMaxElem(std::vector<int> m) {
  int max_elem = m[0];
  for (size_t i = 0; i < m.size(); i++) {
    if (m[i] > max_elem) {
      max_elem = m[i];
    }
  }
  return max_elem;
}

bool vladimirova_j_max_of_vector_elements_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  input_ = std::vector<int>(taskData->inputs_count[0] * taskData->inputs_count[1]);

  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* input_data = reinterpret_cast<int*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i * taskData->inputs_count[1] + j] = input_data[j];
    }
  }
  res = INT_MIN;
  return true;
}

bool vladimirova_j_max_of_vector_elements_seq::TestTaskSequential::validation() {
  internal_order_test();

  return (taskData->outputs_count[0] == 1) && ((taskData->inputs_count[0] > 0) && (taskData->inputs_count[1] > 0));
}

bool vladimirova_j_max_of_vector_elements_seq::TestTaskSequential::run() {
  internal_order_test();

  res = vladimirova_j_max_of_vector_elements_seq::FindMaxElem(input_);
  return true;
}

bool vladimirova_j_max_of_vector_elements_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
