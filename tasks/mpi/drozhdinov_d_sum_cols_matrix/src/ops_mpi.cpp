// Copyright 2023 Nesterov Alexander
#include "mpi/drozhdinov_d_sum_cols_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> drozhdinov_d_sum_cols_matrix_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = (gen() % 100) - 49;
  }
  return vec;
}

int drozhdinov_d_sum_cols_matrix_mpi::makeLinCoords(int x, int y, int xSize) { return y * xSize + x; }

std::vector<int> drozhdinov_d_sum_cols_matrix_mpi::calcMatSumSeq(const std::vector<int>& matrix, int xSize, int ySize,
                                                                 int fromX, int toX) {
  std::vector<int> result;
  for (int x = fromX; x < toX; x++) {
    int columnSum = 0;
    for (int y = 0; y < ySize; y++) {
      int linearizedCoordinate = makeLinCoords(x, y, xSize);
      columnSum += matrix[linearizedCoordinate];
    }
    result.push_back(columnSum);
  }
  return result;
}

bool drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = ptr[i];
  }
  cols = taskData->inputs_count[1];
  rows = taskData->inputs_count[2];
  res = std::vector<int>(cols, 0);
  return true;
}

bool drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = calcMatSumSeq(input_, cols, rows, 0, cols);
  return true;
}

bool drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < cols; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

bool drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    rows = taskData->inputs_count[2];
    cols = taskData->inputs_count[1];
  }
  broadcast(world, cols, 0);
  broadcast(world, rows, 0);
  // fbd nt ncssr dlt
  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  } else {
    input_ = std::vector<int>(cols * rows);
  }
  broadcast(world, input_.data(), cols * rows, 0);
  // Init value for output
  res = std::vector<int>(cols, 0);
  return true;
}

bool drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == taskData->inputs_count[1];
  }
  return true;
}

bool drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int delta = cols / world.size();
  delta += (cols % world.size() == 0) ? 0 : 1;
  int lastCol = std::min(cols, delta * (world.rank() + 1));
  auto localSum = calcMatSumSeq(input_, cols, rows, delta * world.rank(), lastCol);
  localSum.resize(delta);
  if (world.rank() == 0) {
    std::vector<int> localRes(cols + delta * world.size());
    std::vector<int> sizes(world.size(), delta);
    boost::mpi::gatherv(world, localSum.data(), localSum.size(), localRes.data(), sizes, 0);
    localRes.resize(cols);
    res = localRes;
  } else {
    boost::mpi::gatherv(world, localSum.data(), localSum.size(), 0);
  }
  return true;
}

bool drozhdinov_d_sum_cols_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < cols; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}
