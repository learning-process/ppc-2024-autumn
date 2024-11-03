#include "seq/morozov_e_min_val_in_rows_matrix/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define uint unsigned int

using namespace std::chrono_literals;
bool morozov_e_min_val_in_rows_matrix::TestTaskSequential::pre_processing() {
  internal_order_test();
  int n = taskData->inputs_count[0];
  int m = taskData->inputs_count[1];
  matrix_ = std::vector<std::vector<int>>(n, std::vector<int>(m));
  min_val_list_ = std::vector<int>(n);
  for (int i = 0; i < n; ++i) {
    int* input_matrix = reinterpret_cast<int*>(taskData->inputs[i]);
    for (int j = 0; j < m; ++j) {
      matrix_[i][j] = input_matrix[j];
    }
  }
  // std::cout << "HELLO";
  return true;
}
bool morozov_e_min_val_in_rows_matrix::TestTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs.empty() || taskData->outputs.empty() || taskData->inputs_count.size() < 2 ||
      taskData->inputs_count[0] <= 0 || taskData->inputs_count[1] <= 0 || taskData->outputs_count.size() != 1 ||
      taskData->outputs_count[0] != taskData->inputs_count[0])
    return false;
  return true;
}
bool morozov_e_min_val_in_rows_matrix::TestTaskSequential::run() {
  internal_order_test();
  int n = taskData->inputs_count[0];
  int m = taskData->inputs_count[1];
  for (int i = 0; i < n; ++i) {
    int cur_max = matrix_[i][0];
    for (int j = 0; j < m; ++j) {
      cur_max = std::min(cur_max, matrix_[i][j]);
    }
    min_val_list_[i] = cur_max;
  }
  return true;
}
bool morozov_e_min_val_in_rows_matrix::TestTaskSequential::post_processing() {
  internal_order_test();
  int* outputs = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < min_val_list_.size(); i++) {
    outputs[i] = min_val_list_[i];
  }
  return true;
}