// Copyright 2024 Nesterov Alexander
#include "seq/Shurygin_S_max_po_stolbam_matrix/include/ops_seq.hpp"

#include <thread>
using namespace std::chrono_literals;

namespace Shurygin_S_max_po_stolbam_matrix_seq {

bool TestTaskSequential::pre_processing() {
  internal_order_test();
  int rows = taskData->inputs_count[0];
  int columns = taskData->inputs_count[1];
  input_.resize(rows, std::vector<int>(columns));
  for (int i = 0; i < rows; i++) {
    int* input_matrix = reinterpret_cast<int*>(taskData->inputs[i]);
    for (int j = 0; j < columns; j++) {
      input_[i][j] = input_matrix[j];
    }
  }
  res_.resize(columns);
  return true;
}

bool TestTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs.empty() || taskData->outputs.empty()) {
    return false;
  }
  if (taskData->inputs_count.size() < 2 || taskData->inputs_count[0] <= 0 || taskData->inputs_count[1] <= 0) {
    return false;
  }
  if (taskData->outputs_count.size() != 1 || taskData->outputs_count[0] != taskData->inputs_count[1]) {
    return false;
  }
  return true;
}

bool TestTaskSequential::run() {
  internal_order_test();
  for (size_t j = 0; j < input_[0].size(); j++) {
    int max_val = input_[0][j];
    for (size_t i = 1; i < input_.size(); i++) {
      if (input_[i][j] > max_val) {
        max_val = input_[i][j];
      }
    }
    res_[j] = max_val;
  }
  return true;
}

bool TestTaskSequential::post_processing() {
  internal_order_test();
  int* output_matrix = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res_.size(); i++) {
    output_matrix[i] = res_[i];
  }
  return true;
}

std::vector<int> TestTaskSequential::generating_random_vector(int size, int lower_bound, int upper_bound) {
  std::vector<int> v1(size);
  for (auto& num : v1) {
    num = lower_bound + std::rand() % (upper_bound - lower_bound + 1);
  }
  return v1;
}

std::vector<std::vector<int>> TestTaskSequential::generate_random_matrix(int rows, int columns) {
  std::vector<std::vector<int>> matrix1(rows, std::vector<int>(columns));
  for (int i = 0; i < rows; ++i) {
    matrix1[i] = generating_random_vector(columns, 1, 100);
  }
  for (int j = 0; j < columns; ++j) {
    int random_row = std::rand() % rows;
    matrix1[random_row][j] = 200;
  }
  return matrix1;
}
}  // namespace Shurygin_S_max_po_stolbam_matrix_seq
