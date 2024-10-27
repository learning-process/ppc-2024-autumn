// Copyright 2023 Nesterov Alexander
#include "mpi/chizhov_m_max_values_by_columns_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> chizhov_m_max_values_by_columns_matrix_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool chizhov_m_max_values_by_columns_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors

  cols = taskData->inputs_count[1];
  rows = taskData->inputs_count[2];

  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);

  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }

  res_ = std::vector<int>(cols, 0);

  return true;
}

bool chizhov_m_max_values_by_columns_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[1] == 0 || taskData->inputs_count[2] == 0) {
    return false;
  }
  if (taskData->inputs.empty()|| taskData->inputs_count[0] <= 0) {
    return false;
  }
  if (taskData->inputs_count[1] != taskData->outputs_count[0]){
    return false;
  }
  return true;
}

bool chizhov_m_max_values_by_columns_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (int j = 0; j < cols; j++) {
    int maxElement = input_[j];  
    for (int i = 1; i < rows; i++) {
      if (input_[i * cols + j] > maxElement) {
        maxElement = input_[i * cols + j];
      }
    }
    res_[j] = maxElement;
  }
  return true;
}

bool chizhov_m_max_values_by_columns_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < cols; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
  }
  return true;
}

bool chizhov_m_max_values_by_columns_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    cols = taskData->inputs_count[1];
    rows = taskData->inputs_count[2];
  }

  broadcast(world, cols, 0);
  broadcast(world, rows, 0);

  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  } else {
    input_ = std::vector<int>(cols * rows, 0);
  }

  broadcast(world, input_.data(), cols * rows, 0);

  res_ = std::vector<int>(cols, 0);

  return true;
}

bool chizhov_m_max_values_by_columns_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[1] == 0 || taskData->inputs_count[2] == 0) {
      return false;
    }
    if (taskData->inputs.empty() || taskData->inputs_count[0] <= 0) {
      return false;
    }
    if (taskData->inputs_count[1] != taskData->outputs_count[0]) {
      return false;
    }
  }
  return true;
}

bool chizhov_m_max_values_by_columns_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int delta = cols / world.size();
  int extra = cols % world.size();
  if (extra != 0) {
    delta += 1;
  }
  int startCol = delta * world.rank();
  int lastCol = std::min(cols, delta * (world.rank() + 1));
  std::vector<int> localMax;
  for (int j = startCol; j < lastCol; j++) {
    int maxElem = input_[j];
    for (int i = 1; i < rows; i++) {
      int coor = i * cols + j;
      if (input_[coor] > maxElem) {
        maxElem = input_[coor];
      }
    }
    localMax.push_back(maxElem);
  }
  localMax.resize(delta);
  if (world.rank() == 0) {
    std::vector<int> globalRes(cols + delta * world.size());
    std::vector<int> sizes(world.size(), delta);
    boost::mpi::gatherv(world, localMax.data(), localMax.size(), globalRes.data(), sizes, 0);
    globalRes.resize(cols);
    res_ = globalRes;
  } else {
    boost::mpi::gatherv(world, localMax.data(), localMax.size(), 0);
  }
  return true;
}

bool chizhov_m_max_values_by_columns_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < cols; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
    }
  }
  return true;
}
