// Copyright 2023 Nesterov Alexander
#include "mpi/naumov_b_min_colum_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> naumov_b_min_colum_matrix_mpi::getRandomVector(int size) {
  std::vector<int> vec(size);
  for (int& element : vec) {
    element = rand() % 201 - 100;
  }
  return vec;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  input_.resize(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    for (unsigned j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i][j] = tmp_ptr[i * taskData->inputs_count[1] + j];
    }
  }

  res.resize(taskData->inputs_count[1], std::numeric_limits<int>::max());
  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == taskData->inputs_count[1];
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  size_t numRows = input_.size();
  size_t numCols = input_[0].size();

  for (size_t j = 0; j < numCols; j++) {
    res[j] = input_[0][j];
    for (size_t i = 1; i < numRows; i++) {
      res[j] = std::min(res[j], input_[i][j]);
    }
  }

  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int rows = 0;
  int cols = 0;

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
  }

  broadcast(world, rows, 0);
  broadcast(world, cols, 0);

  int delta = rows / world.size();
  int extra = rows % world.size();

  int local_rows = (world.rank() < extra) ? (delta + 1) : delta;
  local_input_.resize(local_rows, std::vector<int>(cols));

  if (world.rank() == 0) {
    input_.resize(rows, std::vector<int>(cols));
    auto* input_matrix = reinterpret_cast<int*>(taskData->inputs[0]);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        input_[i][j] = input_matrix[i * cols + j];
      }
    }

    for (int proc = 1; proc < world.size(); proc++) {
      int start_row = proc * delta + std::min(proc, extra);
      int num_rows = delta + (proc < extra ? 1 : 0);
      for (int r = start_row; r < start_row + num_rows; r++) {
        world.send(proc, 0, input_[r].data(), cols);
      }
    }
  }

  if (world.rank() == 0) {
    std::copy(input_.begin(), input_.begin() + local_rows, local_input_.begin());
  } else {
    for (int r = 0; r < local_rows; r++) {
      world.recv(0, 0, local_input_[r].data(), cols);
    }
  }

  res.resize(cols, std::numeric_limits<int>::max());
  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return (!taskData->inputs.empty() && !taskData->outputs.empty()) && (taskData->inputs_count.size() >= 2) &&
           (taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
           (taskData->outputs_count[0] == taskData->inputs_count[1]);
  }

  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int local_rows = local_input_.size();
  if (local_rows == 0) return true;

  int numCols = local_input_[0].size();
  std::vector<int> local_minima(numCols, std::numeric_limits<int>::max());

  for (int j = 0; j < numCols; ++j) {
    for (int i = 0; i < local_rows; ++i) {
      local_minima[j] = std::min(local_minima[j], local_input_[i][j]);
    }
  }

  if (world.rank() == 0) {
    std::vector<int> all_minima(numCols, std::numeric_limits<int>::max());

    for (int j = 0; j < numCols; ++j) {
      all_minima[j] = local_minima[j];
    }

    for (int proc = 1; proc < world.size(); ++proc) {
      std::vector<int> recv_minima(numCols, std::numeric_limits<int>::max());
      world.recv(proc, 0, recv_minima.data(), numCols);

      for (int j = 0; j < numCols; ++j) {
        all_minima[j] = std::min(all_minima[j], recv_minima[j]);
      }
    }

    res.assign(all_minima.begin(), all_minima.end());
  } else {
    world.send(0, 0, local_minima.data(), numCols);
  }

  return true;
}

bool naumov_b_min_colum_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }

  return true;
}
