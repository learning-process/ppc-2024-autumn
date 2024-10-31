// Copyright 2023 Nesterov Alexander
#include "mpi/gordeva_t_max_val_of_column_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// std::vector<int> gordeva_t_max_val_of_column_matrix_mpi::getRandomVector(int sz) {
//   std::random_device dev;
//   std::mt19937 gen(dev());
//   std::vector<int> vec(sz);
//   for (int i = 0; i < sz; i++) {
//     vec[i] = gen() % 100;
//   }
//   return vec;
// }

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  // Init matrix

  int rows = taskData->inputs_count[0];
  int cols = taskData->inputs_count[1];
  // int* input_matr;
  input_.resize(rows, std::vector<int>(cols));

  for (int i = 0; i < rows; i++) {
    int* input_matr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (int j = 0; j < cols; j++) input_[i][j] = input_matr[j];
  }

  res.resize(cols);

  return true;
}

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  // Check count elements of output
  if ((taskData->inputs.empty() || taskData->outputs.empty()) ||
      (taskData->inputs_count.size() < 2 || taskData->inputs_count[0] <= 0 || taskData->inputs_count[1] <= 0) ||
      (taskData->outputs_count.size() != 1 || taskData->outputs_count[0] != taskData->inputs_count[1]))
    return false;
  return true;
}

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  // int cols = input_[0].size();
  // int rows = input_.size();

  for (size_t i = 0; i < input_[0].size(); i++) {
    int max_el = input_[0][i];
    for (size_t j = 1; j < input_.size(); j++)
      if (input_[j][i] > max_el) max_el = input_[j][i];

    res[i] = max_el;
  }

  return true;
}

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  int* output_matr = reinterpret_cast<int*>(taskData->outputs[0]);

  for (size_t i = 0; i < res.size(); i++) output_matr[i] = res[i];
  return true;
}

std::vector<int> gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::gen_rand_vec(int s, int low, int upp) {
  std::vector<int> v(s);
  for (auto& i : v) i = low + (std::rand() % (upp - low + 1));
  return v;
}

std::vector<std::vector<int>> gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::gen_rand_matr(int rows,
                                                                                                           int cols) {
  std::vector<std::vector<int>> matr(rows, std::vector<int>(cols));

  for (int i = 0; i < rows; ++i) {
    matr[i] = gen_rand_vec(cols, -500, 500);
  }
  for (int j = 0; j < cols; ++j) {
    int row_rand = std::rand() % rows;
    matr[row_rand][j] = 10;
  }
  return matr;
}

// Parallel

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int rows = 0;
  int cols = 0;

  int delta = 0;
  int delta_1 = 0;
  // int* input_matr;

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
  }
  // delta = taskData->inputs_count[0] / world.size();
  // delta_1 = taskData->inputs_count[0] % world.size();

  broadcast(world, rows, 0);
  broadcast(world, cols, 0);

  delta = rows / world.size();
  delta_1 = rows % world.size();

  if (world.rank() == 0) {
    // Init vectors
    input_.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; i++) {
      int* input_matr = reinterpret_cast<int*>(taskData->inputs[i]);
      input_[i].assign(input_matr, input_matr + cols);
    }

    for (int proc = 1; proc < world.size(); proc++) {
      // world.send(proc, 0, input_.data() + proc * delta, delta);
      int row_1 = proc * delta + std::min(proc, delta_1);
      int kol_vo = delta + (proc < delta_1 ? 1 : 0);

      for (int i = row_1; i < row_1 + kol_vo; i++) world.send(proc, 0, input_[i].data(), cols);
    }
  }

  int local_input_rows = delta + (world.rank() < delta_1 ? 1 : 0);
  local_input_.resize(local_input_rows, std::vector<int>(cols));

  if (world.rank() == 0) {
    std::copy(input_.begin(), input_.begin() + local_input_rows, local_input_.begin());
  } else {
    // world.recv(0, 0, local_input_.data(), delta);
    for (int i = 0; i < local_input_rows; i++) world.recv(0, 0, local_input_[i].data(), cols);
  }

  // Init value for output
  res.resize(cols);
  return true;
}

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    // Check count elements of output
    if ((taskData->inputs.empty() || taskData->outputs.empty()) ||
        (taskData->inputs_count.size() < 2 || taskData->inputs_count[0] <= 0 || taskData->inputs_count[1] <= 0) ||
        (taskData->outputs_count.size() != 1 || taskData->outputs_count[0] != taskData->inputs_count[1]))
      return false;
  }
  return true;
}

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  std::vector<int> tmp_max(local_input_[0].size(), INT_MIN);

  for (size_t i = 0; i < local_input_[0].size(); i++) {
    for (size_t j = 0; j < local_input_.size(); j++) {
      tmp_max[i] = std::max(tmp_max[i], local_input_[j][i]);
    }
  }

  if (world.rank() == 0) {
    std::vector<int> max_s(res.size(), INT_MIN);
    std::copy(tmp_max.begin(), tmp_max.end(), max_s.begin());

    for (int proc = 1; proc < world.size(); proc++) {
      std::vector<int> proc_max(res.size());
      world.recv(proc, 0, proc_max.data(), res.size());

      for (size_t i = 0; i < res.size(); i++) {
        max_s[i] = std::max(max_s[i], proc_max[i]);
      }
    }
    std::copy(max_s.begin(), max_s.end(), res.begin());
  } else {
    world.send(0, 0, tmp_max.data(), tmp_max.size());
  }
  return true;
}

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // reinterpret_cast<int*>(taskData->outputs[0]) = res;
    std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
