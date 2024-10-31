#include "mpi/morozov_e_min_val_in_rows_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

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
bool morozov_e_min_val_in_rows_matrix::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  int n = taskData->inputs_count[0];
  int m = taskData->inputs_count[1];
  matrix_ = std::vector<std::vector<int>>(n, std::vector<int>(m));
  min_val_list_ = std::vector<int>(n);
  for (int i = 0; i < n; ++i) {
    int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (int j = 0; j < m; ++j) {
      matrix_[i][j] = tmp_ptr[j];
    }
  }
  // std::cout << "HELLO";
  return true;
}
bool morozov_e_min_val_in_rows_matrix::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->outputs_count.empty() || taskData->inputs_count.empty() ||
      taskData->outputs_count[0] != taskData->inputs_count[0]) {
    return false;
  }
  return true;
}
bool morozov_e_min_val_in_rows_matrix::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < matrix_.size(); ++i) {
    int cur_max = matrix_[i][0];
    for (size_t j = 0; j < matrix_[i].size(); ++j) {
      cur_max = std::min(cur_max, matrix_[i][j]);
    }
    min_val_list_[i] = cur_max;
  }
  return true;
}
bool morozov_e_min_val_in_rows_matrix::TestMPITaskSequential::post_processing() {
  internal_order_test();
  int* outputs = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < min_val_list_.size(); i++) {
    outputs[i] = min_val_list_[i];
  }
  return true;
}
bool morozov_e_min_val_in_rows_matrix::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  int n = 0;
  int m = 0;
  if (world.rank() == 0) {
    n = taskData->inputs_count[0];
    m = taskData->inputs_count[1];
  }
  broadcast(world, n, 0);
  broadcast(world, m, 0);
  int delta = n / world.size();
  int mod = n % world.size();
  if (world.rank() == 0) {
    matrix_ = std::vector<std::vector<int>>(n, std::vector<int>(m));
    for (int i = 0; i < n; ++i) {
      int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      for (int j = 0; j < m; ++j) {
        matrix_[i][j] = tmp_ptr[j];
      }
    }
    for (int i = 1; i < world.size(); i++) {
      int begin_pos = i * delta;
      int cur_count = delta;
      if (i < mod) {
        cur_count++;
      }
      for (int j = begin_pos; j < begin_pos + cur_count; j++) {
        world.send(i, 0, matrix_[j].data(), m);
      }
    }
    int local_rows = delta + (world.rank() < mod ? 1 : 0);
    local_matrix_.resize(local_rows, std::vector<int>(m));
    if (world.rank() == 0) {
      std::copy(matrix_.begin(), matrix_.begin() + local_rows, local_matrix_.begin());
    } else {
      for (int r = 0; r < local_rows; r++) {
        world.recv(0, 0, local_matrix_[r].data(), m);
      }
    }
    min_val_list_.resize(m);
  }
  return true;
}
bool morozov_e_min_val_in_rows_matrix::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0 && (taskData->outputs_count.empty() || taskData->inputs_count.empty() ||
                            taskData->outputs_count[0] != taskData->inputs_count[0])) {
    return false;
  }
  return true;
}
bool morozov_e_min_val_in_rows_matrix::TestMPITaskParallel::run() {
  internal_order_test();
  std::vector<int> res(local_matrix_[0].size(), INT_MAX);
  for (size_t i = 0; i < local_matrix_.size(); i++) {
    for (size_t j = 0; j < local_matrix_[0].size(); j++) {
      res[i] = std::min(res[i], local_matrix_[i][j]);
    }
  }
  if (world.rank() == 0) {
    std::vector<int> cur_min_val_list(min_val_list_.size(), INT_MAX);
    std::copy(res.begin(), res.end(), cur_min_val_list.begin());
    int world_size = world.size();
    for (int i = 1; i < world_size; i++) {
      std::vector<int> res_(min_val_list_.size());
      world.recv(i, 0, res_.data(), min_val_list_.size());
      for (size_t j = 0; j < min_val_list_.size(); j++) {
        cur_min_val_list[j] = std::min(cur_min_val_list[j], res_[j]);
      }
    }
    std::copy(cur_min_val_list.begin(), cur_min_val_list.end(), min_val_list_.begin());
  } else {
    world.send(0, 0, res.data(), res.size());
  }
  return true;
}
bool morozov_e_min_val_in_rows_matrix::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* res = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(min_val_list_.begin(), min_val_list_.end(), res);
  }
  return true;
}