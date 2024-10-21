// Copyright 2023 Nesterov Alexander
#include "mpi/Shurygin_S_max_po_stolbam_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  int rows = taskData->inputs_count[0];
  int cols = taskData->inputs_count[1];
  input_.resize(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; i++) {
    int* input_matrix = reinterpret_cast<int*>(taskData->inputs[i]);
    for (int j = 0; j < cols; j++) {
      input_[i][j] = input_matrix[j];
    }
  }
  res_.resize(cols);
  return true;
}

bool Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential::validation() {
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

bool Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential::run() {
    internal_order_test();
    for (size_t j = 0; j < input_[0].size(); j++) {
      int max_val = input_[0][j];  
      for (size_t i = 1; i < input_.size(); i++) {
        if (input_[i][j] > max_val) {
          max_val = input_[i][j];
        }
      }
      res_[j] = max_val;
    }
    return true;
  }

bool Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential::post_processing() {
    internal_order_test();
    int* output_matrix = reinterpret_cast<int*>(taskData->outputs[0]);
    for (size_t i = 0; i < res_.size(); i++) {
      output_matrix[i] = res_[i];
    }
    return true;
  }



bool Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel::pre_processing() {
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
  if (world.rank() == 0) {
    input_.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; i++) {
      int* input_matrix = reinterpret_cast<int*>(taskData->inputs[i]);
      input_[i].assign(input_matrix, input_matrix + cols);
    }
    for (int proc = 1; proc < world.size(); proc++) {
      int start_row = proc * delta + std::min(proc, extra);
      int num_rows = delta + (proc < extra ? 1 : 0);
      for (int r = start_row; r < start_row + num_rows; r++) {
        world.send(proc, 0, input_[r].data(), cols);
      }
    }
  }
  int local_rows = delta + (world.rank() < extra ? 1 : 0);
  local_input_.resize(local_rows, std::vector<int>(cols));
  if (world.rank() == 0) {
    std::copy(input_.begin(), input_.begin() + local_rows, local_input_.begin());
  } else {
    for (int r = 0; r < local_rows; r++) {
      world.recv(0, 0, local_input_[r].data(), cols);
    }
  }
  res_.resize(cols);  
  return true;
}


bool Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty()) return false;
    if (taskData->inputs_count.size() < 2 || taskData->inputs_count[0] <= 0 || taskData->inputs_count[1] <= 0)
      return false;
    if (taskData->outputs_count.size() != 1 || taskData->outputs_count[0] != taskData->inputs_count[1])
      return false;  
  }
  return true;
}

bool Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  std::vector<int> local_maxes(local_input_[0].size(), INT_MIN);
  for (size_t j = 0; j < local_input_[0].size(); j++) {
    for (size_t i = 0; i < local_input_.size(); i++) {
      local_maxes[j] = std::max(local_maxes[j], local_input_[i][j]);
    }
  }
  if (world.rank() == 0) {
    std::vector<int> global_maxes(res_.size(), INT_MIN);
    std::copy(local_maxes.begin(), local_maxes.end(), global_maxes.begin());
    for (int proc = 1; proc < world.size(); proc++) {
      std::vector<int> proc_maxes(res_.size());
      world.recv(proc, 0, proc_maxes.data(), res_.size());
      for (size_t j = 0; j < res_.size(); j++) {
        global_maxes[j] = std::max(global_maxes[j], proc_maxes[j]);
      }
    }
    std::copy(global_maxes.begin(), global_maxes.end(), res_.begin());
  } else {
    world.send(0, 0, local_maxes.data(), local_maxes.size());
  }
  return true;
}

bool Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* output_matrix = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), output_matrix);
  }
  return true;
}

std::vector<int> Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential::generate_random_vector(
    int size, int lower_bound, int upper_bound) {
  std::vector<int> v1(size);
  for (auto& num : v1) {
    num = lower_bound + std::rand() % (upper_bound - lower_bound + 1);
  }
  return v1;
}

std::vector<std::vector<int>> Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential::generate_random_matrix(
    int rows, int cols) {
  std::vector<std::vector<int>> matrix1(rows, std::vector<int>(cols));
  for (auto& row : matrix1) {
    row = generate_random_vector(cols, 1, 500);
  }
  return matrix1;
}