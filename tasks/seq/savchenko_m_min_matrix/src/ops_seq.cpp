// Copyright 2024 Nesterov Alexander
#include "seq/savchenko_m_min_matrix/include/ops_seq.hpp"

#include <random>
#include <thread>

using namespace std::chrono_literals;

std::vector<int> savchenko_m_min_matrix_seq::getRandomMatrix(int rows, int columns, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  
  // Forming a random matrix
  std::vector<int> matrix(rows * columns);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      matrix[i * columns + j] = min + gen() % (max - min + 1);
    }
  }

  return matrix;
}

bool savchenko_m_min_matrix_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] == 1;
}

bool savchenko_m_min_matrix_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  columns = taskData->inputs_count[0];
  rows = taskData->inputs_count[1];
  matrix = std::vector<int>(rows * columns);

  for (int i = 0; i < rows; ++i) {
    auto* temp = reinterpret_cast<int*>(taskData->inputs[i]);

    for (int j = 0; j < columns; ++j) {
      matrix[i * columns + j] = temp[j];
    }
  }
  res = matrix[0];
  
  return true;
}

bool savchenko_m_min_matrix_seq::TestTaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < matrix.size(); i++) {
    if (matrix[i] < res){
      res = matrix[i];
    }
  }
  return true;
}

bool savchenko_m_min_matrix_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
