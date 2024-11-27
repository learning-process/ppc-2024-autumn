// Copyright 2023 Nesterov Alexander
#include "mpi/korovin_n_min_val_row_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::pre_processing() {
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
  res_.resize(rows);
  return true;
}

bool korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
          (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
          (taskData->outputs_count[0] == taskData->inputs_count[0]));
}

bool korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (size_t i = 0; i < input_.size(); i++) {
    int min_val = input_[i][0];
    for (size_t j = 1; j < input_[i].size(); j++) {
      if (input_[i][j] < min_val) {
        min_val = input_[i][j];
      }
    }
    res_[i] = min_val;
  }
  return true;
}

bool korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  int* output_matrix = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res_.size(); i++) {
    output_matrix[i] = res_[i];
  }
  return true;
}

bool korovin_n_min_val_row_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int rows = taskData->inputs_count[0];
    int cols = taskData->inputs_count[1];

    input_.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; i++) {
      int* input_matrix = reinterpret_cast<int*>(taskData->inputs[i]);
      input_[i].assign(input_matrix, input_matrix + cols);
    }

    res_.resize(rows);
  }

  return true;
}

bool korovin_n_min_val_row_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
            (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
            (taskData->outputs_count[0] == taskData->inputs_count[0]));
  }
  return true;
}

bool korovin_n_min_val_row_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int rows = 0;
  int cols = 0;
  int delta = 0;
  int extra = 0;

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
    delta = rows / world.size();
    extra = rows % world.size();
  }

  broadcast(world, delta, 0);
  broadcast(world, extra, 0);
  broadcast(world, cols, 0);

  if (world.rank() == 0) {
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

  std::vector<int> local_mins(local_input_.size(), INT_MAX);
  for (size_t i = 0; i < local_input_.size(); i++) {
    for (const auto& val : local_input_[i]) {
      local_mins[i] = std::min(local_mins[i], val);
    }
  }

  if (world.rank() == 0) {
    int current_ind = 0;
    std::copy(local_mins.begin(), local_mins.end(), res_.begin());
    current_ind += local_mins.size();
    for (int proc = 1; proc < world.size(); proc++) {
      int loc_size;
      world.recv(proc, 0, &loc_size, 1);
      std::vector<int> loc_res_(loc_size);
      world.recv(proc, 0, loc_res_.data(), loc_size);
      copy(loc_res_.begin(), loc_res_.end(), res_.data() + current_ind);
      current_ind += loc_res_.size();
    }
  } else {
    int loc_res__size = (int)local_mins.size();
    world.send(0, 0, &loc_res__size, 1);
    world.send(0, 0, local_mins.data(), loc_res__size);
  }
  return true;
}

bool korovin_n_min_val_row_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* output_matrix = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), output_matrix);
  }

  return true;
}

std::vector<int> korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_vector(int size, int lower_bound,
                                                                                              int upper_bound) {
  std::vector<int> v1(size);
  for (auto& num : v1) {
    num = lower_bound + std::rand() % (upper_bound - lower_bound + 1);
  }
  return v1;
}

std::vector<std::vector<int>> korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(int rows,
                                                                                                           int cols) {
  std::vector<std::vector<int>> matrix1(rows, std::vector<int>(cols));
  for (auto& row : matrix1) {
    row = generate_rnd_vector(cols, -1000, 1000);
    int rnd_index = std::rand() % cols;
    row[rnd_index] = INT_MIN;
  }
  return matrix1;
}
