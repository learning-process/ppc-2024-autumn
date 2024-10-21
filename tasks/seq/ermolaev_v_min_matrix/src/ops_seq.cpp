// Copyright 2024 Nesterov Alexander
#include "seq/ermolaev_v_min_matrix/include/ops_seq.hpp"

#include <climits>
#include <random>

using namespace std::chrono_literals;

std::vector<int> ermolaev_v_min_matrix_seq::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

std::vector<std::vector<int>> ermolaev_v_min_matrix_seq::getRandomMatrix(int rows, int columns) {
  std::vector<std::vector<int>> vec(rows);

  for (int i = 0; i < rows; i++) {
    vec[i] = ermolaev_v_min_matrix_seq::getRandomVector(columns);
  }
  return vec;
}

bool ermolaev_v_min_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  input_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));

  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }

  // Init value for output
  res_ = INT_MAX;
  return true;
}

bool ermolaev_v_min_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] == 1;
}

bool ermolaev_v_min_matrix_seq::TestTaskSequential::run() {
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

bool ermolaev_v_min_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}
