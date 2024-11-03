#include "mpi/morozov_e_min_val_in_rows_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define uint unsigned int

using namespace std::chrono_literals;
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
    matrix_.resize(n, std::vector<int>(m));
    for (int i = 0; i < n; i++) {
      int* input_matrix = reinterpret_cast<int*>(taskData->inputs[i]);
      matrix_[i].assign(input_matrix, input_matrix + m);
    }
  }
  int cur_n = delta + (world.rank() < mod ? 1 : 0);
  local_matrix_.resize(cur_n, std::vector<int>(m));
  min_val_list_.resize(n);
  return true;
}
bool morozov_e_min_val_in_rows_matrix::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty()) return false;
    if (taskData->inputs_count.size() < 2 || taskData->inputs_count[0] <= 0 || taskData->inputs_count[1] <= 0)
      return false;
    if (taskData->outputs_count.size() != 1 || taskData->outputs_count[0] != taskData->inputs_count[1]) return false;
  }
  return true;
}
bool morozov_e_min_val_in_rows_matrix::TestMPITaskParallel::run() {
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
    for (int proc = 1; proc < world.size(); proc++) {
      int begin = proc * delta + std::min(proc, mod);
      int count = delta + (proc < mod ? 1 : 0);
      for (int r = begin; r < begin + count; r++) {
        world.send(proc, 0, matrix_[r].data(), m);
      }
    }
  }

  int cur_n = delta + (world.rank() < mod ? 1 : 0);

  if (world.rank() == 0) {
    std::copy(matrix_.begin(), matrix_.begin() + cur_n, local_matrix_.begin());
  } else {
    for (int r = 0; r < cur_n; r++) {
      world.recv(0, 0, local_matrix_[r].data(), m);
    }
  }

  std::vector<int> local_mins(local_matrix_.size(), INT_MAX);
  for (size_t i = 0; i < local_matrix_.size(); i++) {
    for (const auto& val : local_matrix_[i]) {
      local_mins[i] = std::min(local_mins[i], val);
    }
  }

  if (world.rank() == 0) {
    int i_cur = 0;
    std::copy(local_mins.begin(), local_mins.end(), min_val_list_.begin());
    i_cur += local_mins.size();
    for (int proc = 1; proc < world.size(); proc++) {
      int loc_size;
      world.recv(proc, 0, &loc_size, 1);
      std::vector<int> loc_res_(loc_size);
      world.recv(proc, 0, loc_res_.data(), loc_size);
      copy(loc_res_.begin(), loc_res_.end(), min_val_list_.data() + i_cur);
      i_cur += loc_res_.size();
    }
  } else {
    int count = (int)local_mins.size();
    world.send(0, 0, &count, 1);
    world.send(0, 0, local_mins.data(), count);
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