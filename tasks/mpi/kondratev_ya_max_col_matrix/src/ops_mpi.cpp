// Copyright 2023 Nesterov Alexander
#include "mpi/kondratev_ya_max_col_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<std::vector<int32_t>> kondratev_ya_max_col_matrix_mpi::getRandomMatrix(uint32_t row, uint32_t col) {
  if (row == 0 || col == 0) {
    throw std::invalid_argument("Args should be greater then zero");
  }

  uint32_t low = -200;
  uint32_t high = 200;

  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<int32_t>> mtrx(row, std::vector<int32_t>(col));
  for (uint32_t i = 0; i < row; i++) {
    for (uint32_t j = 0; j < col; j++) {
      mtrx[i][j] = low + gen() % (high - low + 1);
    }
  }
  return mtrx;
}

void kondratev_ya_max_col_matrix_mpi::insertRefValue(std::vector<std::vector<int32_t>>& mtrx, int32_t ref) {
  std::random_device dev;
  std::mt19937 gen(dev());

  uint32_t ind;
  uint32_t row = mtrx.size();
  uint32_t col = mtrx[0].size();

  for (uint32_t j = 0; j < col; j++) {
    ind = gen() % row;
    mtrx[ind][j] = ref;
  }
}

bool kondratev_ya_max_col_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  uint32_t row = taskData->inputs_count[0];
  uint32_t col = taskData->inputs_count[1];

  std::vector<int32_t*> tmp(row);
  for (uint32_t i = 0; i < row; i++) {
    tmp[i] = reinterpret_cast<int32_t*>(taskData->inputs[i]);
  }

  input_ = std::vector(col, std::vector<int32_t>(row));
  for (uint32_t j = 0; j < col; j++) {
    for (uint32_t i = 0; i < row; i++) {
      input_[j][i] = tmp[i][j];
    }
  }
  res_ = std::vector<int32_t>(col);

  return true;
}

bool kondratev_ya_max_col_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == taskData->inputs_count[1];
}

bool kondratev_ya_max_col_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (uint32_t i = 0; i < input_.size(); i++) {
    res_[i] = *std::max_element(input_[i].begin(), input_[i].end());
  }

  return true;
}

bool kondratev_ya_max_col_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  auto* output_matrix = reinterpret_cast<int32_t*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), output_matrix);

  return true;
}

bool kondratev_ya_max_col_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  uint32_t rows = 0;
  uint32_t cols = 0;
  uint32_t step = 0;
  uint32_t recvSize = 0;

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
    step = std::ceil((double)cols / world.size());
  }

  broadcast(world, rows, 0);
  broadcast(world, cols, 0);
  broadcast(world, step, 0);

  if (world.rank() == 0) {
    std::vector<int32_t*> tmp(rows);
    for (uint32_t i = 0; i < rows; i++) {
      tmp[i] = reinterpret_cast<int32_t*>(taskData->inputs[i]);
    }

    input_ = std::vector(cols, std::vector<int32_t>(rows));
    for (uint32_t j = 0; j < cols; j++) {
      for (uint32_t i = 0; i < rows; i++) {
        input_[j][i] = tmp[i][j];
      }
    }
    res_ = std::vector<int32_t>(cols);

    uint32_t worldSize = world.size();

    auto it = input_.begin() + step;
    auto end = input_.end();
    for (uint32_t i = 1; i < worldSize && it < end; i++) {
      recvSize = std::min(step, static_cast<uint32_t>(end - it));
      world.send(i, 0, recvSize);

      for (uint32_t j = 0; j < recvSize; j++) {
        world.send(i, 0, *(it++));
      }
    }
  }

  if (world.rank() == 0) {
    local_input_ = std::vector(step, std::vector<int32_t>(rows));
    std::copy(input_.begin(), input_.begin() + step, local_input_.begin());
  } else {
    world.recv(0, 0, recvSize);
    local_input_.resize(recvSize);
    for (uint32_t i = 0; i < recvSize; i++) {
      world.recv(0, 0, local_input_[i]);
    }
  }

  return true;
}

bool kondratev_ya_max_col_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return taskData->outputs_count[0] == taskData->inputs_count[1];
  }
  return true;
}

bool kondratev_ya_max_col_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  std::vector<int32_t> loc_max(local_input_.size());
  for (size_t i = 0; i < loc_max.size(); i++) {
    loc_max[i] = *std::max_element(local_input_[i].begin(), local_input_[i].end());
  }

  gather(world, loc_max.data(), loc_max.size(), res_, 0);

  return true;
}

bool kondratev_ya_max_col_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* output_matrix = reinterpret_cast<uint32_t*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), output_matrix);
  }

  return true;
}
