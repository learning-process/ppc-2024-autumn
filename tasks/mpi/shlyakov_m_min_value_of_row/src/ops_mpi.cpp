// Copyright 2023 Nesterov Alexander
#include "mpi/shlyakov_m_min_value_of_row/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace std::chrono_literals;

bool shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  size_t sz_row = taskData->inputs_count[0];
  size_t sz_col = taskData->inputs_count[1];
  input_.resize(sz_row, std::vector<int>(sz_col));

  for (size_t i = 0; i < sz_row; i++) {
    auto* matr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (size_t j = 0; j < sz_col; j++) {
      input_[i][j] = matr[j];
    }
  }
  res_.resize(sz_row);

  return true;
}

bool shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  if (((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
       (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0)) &&
      (taskData->outputs_count[0] == taskData->inputs_count[0]))
    return (true);

  return (false);
}

bool shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  int min;
  size_t sz_row = input_.size();
  size_t sz_col = input_[0].size();

  for (size_t i = 0; i < sz_row; i++) {
    min = input_[i][0];
    for (size_t j = 1; j < sz_col; j++) {
      if (input_[i][j] < min) {
        min = input_[i][j];
      }
    }
    res_[i] = min;
  }

  return true;
}

bool shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  int* result = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res_.size(); i++) {
    result[i] = res_[i];
  }

  return true;
}

bool shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
         (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0)) &&
        (taskData->outputs_count[0] == taskData->inputs_count[0]))
      return (true);
    return (false);
  }

  return true;
}

bool shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int sz_row = 0;
  int sz_col = 0;

  if (world.rank() == 0) {
    sz_row = taskData->inputs_count[0];
    sz_col = taskData->inputs_count[1];
  }

  broadcast(world, sz_row, 0);
  broadcast(world, sz_col, 0);

  int del = sz_row / world.size();
  int ex = sz_row % world.size();

  if (world.rank() == 0) {
    input_.resize(sz_row, std::vector<int>(sz_col));

    for (int i = 0; i < sz_row; i++) {
      int* matr = reinterpret_cast<int*>(taskData->inputs[i]);
      input_[i].assign(matr, matr + sz_col);
    }

    for (int proc = 1; proc < world.size(); proc++) {
      int start_row = proc * del + std::min(proc, ex);
      int num_rows = del + (proc < ex ? 1 : 0);
      for (int r = start_row; r < start_row + num_rows; r++) world.send(proc, 0, input_[r].data(), sz_col);
    }
  }

  int local_rows = del + (world.rank() < ex ? 1 : 0);

  local_input_.resize(local_rows, std::vector<int>(sz_col));

  if (world.rank() == 0)
    std::copy(input_.begin(), input_.begin() + local_rows, local_input_.begin());
  else {
    for (int r = 0; r < local_rows; r++) world.recv(0, 0, local_input_[r].data(), sz_col);
  }

  res_.resize(sz_row);

  std::vector<int> local_mins(local_input_.size(), INT_MAX);
  for (size_t i = 0; i < local_input_.size(); i++) {
    for (const auto& val : local_input_[i]) {
      local_mins[i] = std::min(local_mins[i], val);
    }
  }

  if (world.rank() == 0) {
    int c_ind = 0;
    std::copy(local_mins.begin(), local_mins.end(), res_.begin());
    c_ind += local_mins.size();

    for (int proc = 1; proc < world.size(); proc++) {
      int local_sz;
      world.recv(proc, 0, &local_sz, 1);
      std::vector<int> loc_res_(local_sz);
      world.recv(proc, 0, loc_res_.data(), local_sz);
      copy(loc_res_.begin(), loc_res_.end(), res_.data() + c_ind);
      c_ind += loc_res_.size();
    }
  } else {
    int loc_res__size = (int)local_mins.size();
    world.send(0, 0, &loc_res__size, 1);
    world.send(0, 0, local_mins.data(), loc_res__size);
  }

  return true;
}

bool shlyakov_m_min_value_of_row_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* output_matrix = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), output_matrix);
  }

  return true;
}

std::vector<std::vector<int>> shlyakov_m_min_value_of_row_mpi::TestMPITaskSequential::get_random_matr(int sz_row,
                                                                                                      int sz_col) {
  std::vector<int> rand_vec(sz_row);
  std::vector<std::vector<int>> rand_matr(sz_row, std::vector<int>(sz_col));

  for (auto& row : rand_matr) {
    for (auto& el : rand_vec) el = std::rand() % (1001) - 500;
    row = rand_vec;
    row[std::rand() % sz_col] = INT_MIN;
  }

  return rand_matr;
}