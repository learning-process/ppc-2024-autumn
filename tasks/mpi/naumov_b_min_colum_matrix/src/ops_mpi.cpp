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

std::vector<std::vector<int>> naumov_b_min_colum_matrix_mpi::generate_rnd_matrix(int rows, int columns) {
  std::vector<std::vector<int>> matrix(rows);
  for (int i = 0; i < rows; ++i) {
    matrix[i] = getRandomVector(columns);
  }
  return matrix;
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

  if (world.rank() == 0) {
    input_ = std::vector(taskData->inputs_count[1] * taskData->inputs_count[0], 0);
    auto* temp = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      for (unsigned j = 0; j < taskData->inputs_count[1]; j++) {
        input_[i + j * taskData->inputs_count[0]] = temp[j + i * taskData->inputs_count[1]];
      }
    }
  }

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
  int rows = 0;
  int cols = 0;

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
  }

  broadcast(world, rows, 0);
  broadcast(world, cols, 0);

  int delta = cols / world.size();
  int extra = cols % world.size();

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      int send_size = delta * rows;
      world.send(proc, 0, input_.data() + (proc * delta + extra) * rows, send_size);
    }

    local_vector_ = std::vector<int>(input_.begin(), input_.begin() + (delta + extra) * rows);
  } else {
    local_vector_ = std::vector<int>(delta * rows);
    world.recv(0, 0, local_vector_.data(), delta * rows);
  }

  std::vector<int> local_res(delta + ((world.rank() == 0) ? extra : 0), std::numeric_limits<int>::max());

  for (auto i = 0u; i < local_res.size(); i++) {
    for (int j = 0; j < rows; j++) {
      local_res[i] = std::min(local_res[i], local_vector_[j + rows * i]);
    }
  }

  if (world.rank() == 0) {
    std::vector<int> temp(delta, 0);
    res.insert(res.end(), local_res.begin(), local_res.end());
    for (int i = 1; i < world.size(); i++) {
      world.recv(i, 0, temp.data(), delta);
      res.insert(res.end(), temp.begin(), temp.end());
    }
  } else {
    world.send(0, 0, local_res.data(), delta);
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
