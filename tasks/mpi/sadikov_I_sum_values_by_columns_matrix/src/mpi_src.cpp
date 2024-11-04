#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "mpi/sadikov_I_sum_values_by_columns_matrix/include/ops_mpi.h"

using namespace std::literals::chrono_literals;

sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask::MPITask(std::shared_ptr<ppc::core::TaskData> td)
    : Task(std::move(td)) {}

bool sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask::validation() {
  internal_order_test();
  return taskData->inputs_count[1] == taskData->outputs_count[0];
}

bool sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask::pre_processing() {
  internal_order_test();
  rows_count = static_cast<size_t>(taskData->inputs_count[0]);
  columns_count = static_cast<size_t>(taskData->inputs_count[1]);
  auto *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
  matrix.reserve(columns_count * rows_count);
  for (size_t i = 0; i < columns_count; ++i) {
    for (size_t j = 0; j < rows_count; ++j) {
      matrix.emplace_back(tmp_ptr[j * columns_count + i]);
    }
  }
  sum = std::vector<int>(columns_count);
  return true;
}

bool sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask::run() {
  internal_order_test();
  calculate(columns_count);
  return true;
}

bool sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < columns_count; ++i) {
    reinterpret_cast<int *>(taskData->outputs[0])[i] = sum[i];
  }
  return true;
}

void sadikov_I_Sum_values_by_columns_matrix_mpi::MPITask::calculate(size_t size) {
  for (size_t i = 0; i < size; ++i) {
    sum[i] = std::accumulate(matrix.begin() + i * rows_count, matrix.begin() + (i + 1) * rows_count, 0);
  }
}

sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel::MPITaskParallel(std::shared_ptr<ppc::core::TaskData> td)
    : Task(std::move(td)) {}

bool sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[1] == taskData->outputs_count[0];
  }
  return true;
}

bool sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    rows_count = static_cast<size_t>(taskData->inputs_count[0]);
    columns_count = static_cast<size_t>(taskData->inputs_count[1]);
    delta = columns_count / world.size();
    last_column = columns_count % world.size();
    matrix.reserve(columns_count * rows_count);
    int *tmp_ptr = reinterpret_cast<int *>(taskData->inputs[0]);
    for (size_t i = 0; i < columns_count; ++i) {
      for (size_t j = 0; j < rows_count; ++j) {
        matrix.emplace_back(tmp_ptr[j * columns_count + i]);
      }
    }
  }
  return true;
}

bool sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel::run() {
  internal_order_test();
  broadcast(world, rows_count, 0);
  broadcast(world, columns_count, 0);
  broadcast(world, delta, 0);
  broadcast(world, last_column, 0);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      if (proc != world.size() - 1 && delta != 0) {
        world.send(proc, 0, matrix.data() + proc * rows_count * delta, rows_count * delta);
      }
      if (proc == world.size() - 1 && delta != 0) {
        world.send(proc, 0, matrix.data() + proc * rows_count * (delta), rows_count * (delta + last_column));
      }
    }
  }
  if (delta != 0) {
    local_input = (world.rank() != world.size() - 1) ? std::vector<int>(rows_count * delta)
                                                     : std::vector<int>(rows_count * (delta + last_column));
  } else {
    local_input = std::vector<int>(matrix.begin(), matrix.end());
  }
  if (world.rank() == 0 && delta != 0) {
    local_input = std::vector<int>(matrix.begin(), matrix.begin() + rows_count * delta);

  } else if (world.rank() > 0 && delta != 0) {
    world.recv(0, 0, local_input.data(),
               (world.rank() != world.size() - 1) ? rows_count * delta : rows_count * (delta + last_column));
  }
  size_t size = delta != 0 ? local_input.size() / rows_count : local_input.size();
  std::vector<int> intermediate_res;
  if (delta != 0) {
    intermediate_res = calculate(size);
  }
  if (world.rank() == 0 && delta == 0 && !matrix.empty()) {
    intermediate_res.emplace_back(std::accumulate(local_input.begin(), local_input.end(), 0));
  }
  if (world.rank() == 0) {
    std::vector<int> localRes(columns_count);
    std::vector<int> sizes(world.size(), delta);
    if (delta == 0 && !matrix.empty()) {
      sizes.front() = 1;
    } else if (delta != 0 && !matrix.empty()) {
      sizes.back() = delta + last_column;
    }
    boost::mpi::gatherv(world, intermediate_res, localRes.data(), sizes, 0);
    sum = localRes;
  } else {
    boost::mpi::gatherv(world, intermediate_res, 0);
  }
  return true;
}

bool sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < columns_count; ++i) {
      reinterpret_cast<int *>(taskData->outputs[0])[i] = sum[i];
    }
  }
  return true;
}

std::vector<int> sadikov_I_Sum_values_by_columns_matrix_mpi::MPITaskParallel::calculate(size_t size) {
  std::vector<int> in(size);
  for (size_t i = 0; i < size; ++i) {
    in[i] = std::accumulate(local_input.begin() + i * rows_count, local_input.begin() + (i + 1) * rows_count, 0);
  }
  return in;
}
