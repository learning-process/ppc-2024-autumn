// Copyright 2023 Nesterov Alexander
#include "mpi/koshkin_n_sum_values_by_columns_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> koshkin_n_sum_values_by_columns_matrix_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output

  rows = taskData->inputs_count[0];
  columns = taskData->inputs_count[1];

  // TaskData
  input_.resize(rows, std::vector<int>(columns));

  uint8_t* inputMatrix = taskData->inputs[0];
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      input_[i][j] = reinterpret_cast<int*>(inputMatrix)[i * columns + j];
    }
  }
  res.resize(columns, 0);  // sumColumns
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
          (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
          taskData->inputs_count[1] == taskData->outputs_count[0]);
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (int j = 0; j < columns; ++j) {
    res[j] = 0;
    for (int i = 0; i < rows; ++i) {
      res[j] += input_[i][j];
    }
  }
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  uint8_t* outputSums = taskData->outputs[0];
  for (int j = 0; j < columns; ++j) {
    reinterpret_cast<int*>(outputSums)[j] = res[j];
  }
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  // Init value for input and output

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    columns = taskData->inputs_count[1];
  }

  // TaskData
  input_.resize(rows, std::vector<int>(columns));

  if (world.rank() == 0) {
    uint8_t* inputMatrix = taskData->inputs[0];
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < columns; ++j) {
        input_[i][j] = reinterpret_cast<int*>(inputMatrix)[i * columns + j];
      }
    }
  }
  res.resize(columns, 0);  // sumColumns
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return ((taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
            taskData->inputs_count[1] == taskData->outputs_count[0]);
            
  }
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int j = 0; j < columns; ++j) {
      res[j] = 0;
      for (int i = 0; i < rows; ++i) {
        res[j] += input_[i][j];
      }
    }
  }
  return true;
}

bool koshkin_n_sum_values_by_columns_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    uint8_t* outputSums = taskData->outputs[0];
    for (int j = 0; j < columns; ++j) {
      reinterpret_cast<int*>(outputSums)[j] = res[j];
    }
  }
  return true;
}