#include "mpi/vavilov_v_min_elements_in_columns_of_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  int rows = taskData->inputs_count[0];
  int cols = taskData->inputs_count[1];

  input_.resize(rows, std::vector<int>(cols));

  for (int i = 0; i < rows; i++) {
    int* input_row = reinterpret_cast<int*>(taskData->inputs[i]);
    for (int j = 0; j < cols; j++) {
      input_[i][j] = input_row[j];
    }
  }
  res_.resize(rows);
  return true;
}

bool vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
          (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0) &&
          (taskData->outputs_count[0] == taskData->inputs_count[1]));
}

bool vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  res_.resize(input_[0].size());

  for (size_t i = 0; i < input_[0].size(); i++) {
    int min = input_[0][i];
    for (size_t j = 1; j < input_.size(); j++) {
      if (input_[j][i] < min) {
        min = input_[j][i];
      }
    }
    res_[i] = min;
  }
  return true;
}

bool vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  int* output_matr = reinterpret_cast<int*>(taskData->outputs[0]);
  for (size_t i = 0; i < res_.size(); i++) {
    output_matr[i] = res_[i];
  }
  return true;
}

bool vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int rows = 0;
  int cols = 0;
  int delta_1;
  int delta_2;

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
  }

  broadcast(world, rows, 0);
  broadcast(world, cols, 0);

  delta_1 = rows / world.size();
  delta_2 = rows % world.size();

  if (world.rank() == 0) {
    input_.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; i++) {
      int* input_matr = reinterpret_cast<int*>(taskData->inputs[i]);
      input_[i].assign(input_matr, input_matr + cols);
    }

    for (int proc = 1; proc < world.size(); proc++) {
      int start_row = proc * delta_1 + std::min(proc, delta_2);
      int counts = delta_1 + (proc < delta_2 ? 1 : 0);
      for (int i = start_row; i < start_row + counts; i++) {
        world.send(proc, 0, input_[i].data(), cols);
      }
    }
  }

  int local_rows = delta_1 + (world.rank() < delta_2 ? 1 : 0);

  local_input_.resize(local_rows, std::vector<int>(cols));

  if (world.rank() == 0) {
    std::copy(input_.begin(), input_.begin() + local_rows, local_input_.begin());
  } else {
    for (int i = 0; i < local_rows; i++) {
      world.recv(0, 0, local_input_[i].data(), cols);
    }
  }

  res_.resize(cols);
  return true;
}

bool vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
           (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0) &&
           (taskData->outputs_count[0] == taskData->inputs_count[1]));
  }
  return true;
}

bool vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  std::vector<int> tmp_min(local_input_[0].size(), INT_MAX);
  for (size_t i = 0; i < local_input_[0].size(); i++) {
    for (size_t j = 0; j < local_input_.size(); j++) {
      tmp_min[i] = std::min(tmp_min[i], local_input_[j][i]);
    }
  }

  if (world.rank() == 0) {
    std::vector<int> min_s(res_.size(), INT_MAX);
    std::copy(tmp_min.begin(), tmp_min.end(), min_s.begin());

    for (int proc = 1; proc < world.size(); proc++) {
      std::vector<int> proc_min(res_.size());
      world.recv(proc, 0, proc_min.data(), res_.size());

      for (size_t i = 0; i < res_.size(); i++) {
        min_s[i] = std::min(min_s[i], proc_min[i]);
      }
    }
    std::copy(min_s.begin(), min_s.end(), res_.begin());
  } else {
    world.send(0, 0, tmp_min.data(), tmp_min.size());
  }
  return true;
}

bool vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* output_matr = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), output_matr);
  }

  return true;
}

std::vector<int> vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential::generate_rand_vec(
    int size, int lower_bound, int upper_bound) {
  std::vector<int> vec(size);
  for (auto& n : vec) {
    n = lower_bound + std::rand() % (upper_bound - lower_bound + 1);
  }
  return vec;
}

std::vector<std::vector<int>>
vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential::generate_rand_matr(int rows, int cols) {
  std::vector<std::vector<int>> matr(rows, std::vector<int>(cols));
  for (int i = 0; i < rows; i++) {
    matr[i] = generate_rand_vec(cols, -1000, 1000);
  }
  for (int j = 0; j < cols; j++) {
    int r_row = std::rand() % rows;
    matr[r_row][j] = INT_MIN;
  }
  return matr;
}
