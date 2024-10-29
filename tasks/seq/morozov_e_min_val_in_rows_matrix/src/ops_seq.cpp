#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "seq/morozov_e_min_val_in_rows_matrix/include/ops_seq.hpp"

#define uint unsigned int

using namespace std::chrono_literals;
std::vector<std::vector<int>> morozov_e_min_val_in_rows_matrix::getRandomMatrix(int n, int m) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(-100, 100);

  // Создаем матрицу
  std::vector<std::vector<int>> matrix(n, std::vector<int>(m));

  // Заполняем матрицу случайными значениями
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      matrix[i][j] = dis(gen);
    }
  }
  //matrix = {{1, 2, 3}, {1, 2, 3}, {1, 2, 1}};
  return matrix;
}
std::vector<int> morozov_e_min_val_in_rows_matrix::minValInRowsMatrix(const std::vector<std::vector<int>>& matrix) {
  std::vector<int> res;
  int n = matrix.size();
  int m = matrix[0].size();
  for (int i = 0; i < n; ++i) {
    int cur_min = matrix[i][0];
    for (int j = 1; j < m; ++j) {
      cur_min = std::min(cur_min, matrix[i][j]);
    }
    res.push_back(cur_min);
  }
  return res;
}
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
  if (taskData->outputs_count.empty() || taskData->inputs_count.empty() ||
      taskData->outputs_count[0] != taskData->inputs_count[0]) {
    return false;
  }
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