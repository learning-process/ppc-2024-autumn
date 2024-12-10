#include "mpi/gordeva_t_max_val_of_column_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <climits>
#include <functional>
#include <random>
#include <thread>
#include <vector>

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  int rows = taskData->inputs_count[0];
  int cols = taskData->inputs_count[1];

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

  if (taskData->inputs.empty() || taskData->outputs.empty()) return false;
  if (taskData->inputs_count[0] <= 0 || taskData->inputs_count[1] <= 0) return false;
  if (taskData->outputs_count.size() != 1) return false;
  if (taskData->inputs_count.size() < 2) return false;
  if (taskData->outputs_count[0] != taskData->inputs_count[1]) return false;

  return true;
}

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();
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

  std::copy(res.begin(), res.end(), output_matr);
  return true;
}

std::vector<int> gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::rand_vec(int s, int down, int upp) {
  std::vector<int> v(s);
  for (auto& i : v) i = down + (std::rand() % (upp - down + 1));
  return v;
}

std::vector<std::vector<int>> gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::rand_matr(int rows,
                                                                                                       int cols) {
  std::vector<std::vector<int>> matr(rows, std::vector<int>(cols));

  for (int i = 0; i < rows; ++i) {
    matr[i] = rand_vec(cols, -500, 500);
  }
  for (int j = 0; j < cols; ++j) {
    int row_rand = std::rand() % rows;
    matr[row_rand][j] = 10;
  }
  return matr;
}

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  return true;
}

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty()) return false;
    if (taskData->inputs_count[0] <= 0 || taskData->inputs_count[1] <= 0) return false;
    if (taskData->outputs_count.size() != 1) return false;
    if (taskData->inputs_count.size() < 2) return false;
    if (taskData->outputs_count[0] != taskData->inputs_count[1]) return false;
  }
  return true;
}

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  size_t delta = 0;
  size_t delta_1 = 0;
  size_t local_cols = 0;

  if (world.rank() == 0) {
    size_t rows = taskData->inputs_count[0];
    size_t cols = taskData->inputs_count[1];

    delta = rows / world.size();
    delta_1 = rows % world.size();

    boost::mpi::broadcast(world, delta, 0);
    boost::mpi::broadcast(world, delta_1, 0);

    input_.resize(rows, std::vector<int>(cols));
    for (size_t i = 0; i < rows; i++) {
      int* input_matr = reinterpret_cast<int*>(taskData->inputs[i]);
      input_[i].assign(input_matr, input_matr + cols);
    }

    for (int proc = 1; proc < world.size(); ++proc) {
      size_t start_row = (proc * delta) + std::min(static_cast<size_t>(proc), delta_1);
      size_t rows_to_send = delta + ((static_cast<size_t>(proc) < delta_1) ? 1 : 0);

      world.send(proc, 0, cols);

      for (size_t i = 0; i < rows_to_send; ++i) {
        world.send(proc, 0, input_[start_row + i]);
      }
    }

    size_t local_input_rows = delta + ((static_cast<size_t>(world.rank()) < delta_1) ? 1 : 0);
    local_cols = cols;
    local_input_.assign(input_.begin(), std::next(input_.begin(), static_cast<std::ptrdiff_t>(local_input_rows)));
  } else {
    boost::mpi::broadcast(world, delta, 0);
    boost::mpi::broadcast(world, delta_1, 0);

    size_t local_input_rows = delta + (static_cast<size_t>(world.rank()) < delta_1 ? 1 : 0);

    world.recv(0, 0, local_cols);

    local_input_.resize(local_input_rows, std::vector<int>(local_cols));
    for (size_t i = 0; i < local_input_rows; ++i) {
      world.recv(0, 0, local_input_[i]);
    }
  }

  res.resize(local_cols);
  std::vector<int> tmp_max(local_cols, INT_MIN);

  for (size_t i = 0; i < local_cols; ++i) {
    for (size_t j = 0; j < local_input_.size(); ++j) {
      tmp_max[i] = std::max(tmp_max[i], local_input_[j][i]);
    }
  }

  if (world.rank() == 0) {
    std::vector<int> max_s(local_cols, INT_MIN);
    std::copy(tmp_max.begin(), tmp_max.end(), max_s.begin());

    for (int proc = 1; proc < world.size(); ++proc) {
      std::vector<int> proc_max(local_cols);
      world.recv(proc, 0, proc_max);

      for (size_t i = 0; i < local_cols; ++i) {
        max_s[i] = std::max(max_s[i], proc_max[i]);
      }
    }
    res = max_s;
  } else {
    world.send(0, 0, tmp_max);
  }

  return true;
}

bool gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
