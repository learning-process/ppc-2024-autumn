// Copyright 2023 Nesterov Alexander
#include "mpi/gnitienko_k_sum_values_by_rows_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> gnitienko_k_sum_row_mpi::SumByRowMPISeq::mainFunc() {
  for (int i = 0; i < rows; ++i) {
    int sum = 0;
    for (int j = 0; j < cols; ++j) {
      sum += input_[i * cols + j];
    }
    res[i] = sum;
  }
  return res;
}

bool gnitienko_k_sum_row_mpi::SumByRowMPISeq::pre_processing() {
  internal_order_test();

  rows = taskData->inputs_count[0];
  cols = taskData->inputs_count[1];

  input_.resize(rows * cols);
  auto* ptr = reinterpret_cast<int*>(taskData->inputs[0]);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      input_[i * cols + j] = ptr[i * cols + j];
    }
  }

  res = std::vector<int>(rows, 0);

  return true;
}

bool gnitienko_k_sum_row_mpi::SumByRowMPISeq::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs_count.size() == 2 && taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 &&
          taskData->outputs_count[0] == taskData->inputs_count[0]);
}

bool gnitienko_k_sum_row_mpi::SumByRowMPISeq::run() {
  internal_order_test();
  mainFunc();
  return true;
}


bool gnitienko_k_sum_row_mpi::SumByRowMPISeq::post_processing() {
  internal_order_test();
  // reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  memcpy(taskData->outputs[0], res.data(), rows * sizeof(int));
  return true;
}

// Parallel

bool gnitienko_k_sum_row_mpi::SumByRowMPIParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
  }
  boost::mpi::broadcast(world, rows, 0);
  boost::mpi::broadcast(world, cols, 0);

  if (world.rank() == 0) {
    input_.resize(rows*cols);
    auto* ptr = reinterpret_cast<int*>(taskData->inputs[0]);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        input_[i * cols + j] = ptr[i * cols + j];
      }
    }
  } else {
    input_.resize(rows*cols);
  }

  boost::mpi::broadcast(world, input_.data(), rows*cols, 0);

  return true;
}

bool gnitienko_k_sum_row_mpi::SumByRowMPIParallel::validation() {
  internal_order_test();
  if (world.rank()==0)
    return (taskData->inputs_count.size() == 2 && taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 &&
          taskData->outputs_count[0] == taskData->inputs_count[0]);
  return true;
}

std::vector<int> gnitienko_k_sum_row_mpi::SumByRowMPIParallel::mainFunc(int startRow, int LastRow) {
  std::vector<int> result;
  for (int i = startRow; i < LastRow; i++) {
    int sum_by_row = 0;
    for (int j = 0; j < cols; j++) {
      sum_by_row += input_[i * cols + j];
    }
    result.push_back(sum_by_row);
  }
  return result;
}

bool gnitienko_k_sum_row_mpi::SumByRowMPIParallel::run() {
  internal_order_test();
  
  int rows_per_process = rows / world.size();
  int extra_rows = rows % world.size();
  if (extra_rows != 0)
    rows_per_process += 1;
  int process_last_row = std::min(rows, rows_per_process * (world.rank() + 1));
  std::vector<int> local_sum = mainFunc(rows_per_process * world.rank(), process_last_row);
  local_sum.resize(rows_per_process);

  if (world.rank() == 0) {
    std::vector<int> local_res(rows + rows_per_process * world.size());
    std::vector<int> sizes(world.size(), rows_per_process);
    boost::mpi::gatherv(world, local_sum.data(), local_sum.size(), local_res.data(), sizes, 0);
    local_res.resize(rows);
    res = local_res;
  } else {
    boost::mpi::gatherv(world, local_sum.data(), local_sum.size(), 0);
  }

  return true;
}

bool gnitienko_k_sum_row_mpi::SumByRowMPIParallel::post_processing() {
  internal_order_test();
  
  if (world.rank() == 0) {
    std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }

  return true;
}
