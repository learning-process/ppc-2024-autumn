#include "mpi/borisov_s_sum_of_rows/include/ops_mpi.hpp"

#include <algorithm>
#include <vector>

using namespace std::chrono_literals;

bool borisov_s_sum_of_rows::SumOfRowsTaskSequential ::pre_processing() {
  internal_order_test();

  size_t rows = taskData->inputs_count[0];
  size_t cols = taskData->inputs_count[1];

  if (rows > 0 && cols > 0) {
    matrix_.resize(rows, std::vector<int>(cols));
    int* data = reinterpret_cast<int*>(taskData->inputs[0]);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        matrix_[i][j] = data[(i * cols) + j];
      }
    }
  }

  row_sums_.resize(rows, 0);
  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskSequential::validation() {
  internal_order_test();

  if (taskData->outputs_count[0] != taskData->inputs_count[0]) {
    return false;
  }

  size_t cols = taskData->inputs_count.size() > 1 ? taskData->inputs_count[1] : 0;
  if (cols <= 0) {
    return false;
  }

  if (taskData->inputs[0] == nullptr || taskData->outputs[0] == nullptr) {
    return false;
  }

  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskSequential ::run() {
  internal_order_test();

  if (!matrix_.empty() && !matrix_[0].empty()) {
    for (size_t i = 0; i < matrix_.size(); i++) {
      int row_sum = 0;
      for (size_t j = 0; j < matrix_[i].size(); j++) {
        row_sum += matrix_[i][j];
      }
      row_sums_[i] = row_sum;
    }
  }
  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskSequential ::post_processing() {
  internal_order_test();

  if (!row_sums_.empty()) {
    int* out = reinterpret_cast<int*>(taskData->outputs[0]);
    for (size_t i = 0; i < row_sums_.size(); i++) {
      out[i] = row_sums_[i];
    }
  }
  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskParallel::pre_processing() {
  internal_order_test();

  loc_row_sums_.clear();
  loc_matrix_.clear();

  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskParallel::validation() {
  internal_order_test();

  bool is_valid = true;

  size_t inputs_count_size = taskData->inputs_count.size();
  size_t outputs_count_size = taskData->outputs_count.size();

  if (world.rank() == 0) {
    if (inputs_count_size < 2 || outputs_count_size < 1) {
      is_valid = false;
    } else {
      if (taskData->outputs_count[0] != taskData->inputs_count[0]) {
        is_valid = false;
      }

      size_t cols = taskData->inputs_count[1];
      if (cols == 0) {
        is_valid = false;
      }

      if (taskData->inputs[0] == nullptr || taskData->outputs[0] == nullptr) {
        is_valid = false;
      }
    }
  }

  boost::mpi::broadcast(world, is_valid, 0);

  return is_valid;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskParallel::run() {
  internal_order_test();

  size_t rows = 0;
  size_t cols = 0;

  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    cols = taskData->inputs_count[1];
  }

  boost::mpi::broadcast(world, rows, 0);
  boost::mpi::broadcast(world, cols, 0);

  size_t base_rows_per_proc = rows / world.size();
  int remainder_rows = static_cast<int>(rows % world.size());

  size_t local_rows = base_rows_per_proc + (world.rank() < remainder_rows ? 1 : 0);

  std::vector<int> sendcounts(world.size());
  std::vector<int> displs(world.size());

  if (world.rank() == 0) {
    size_t offset = 0;
    for (int i = 0; i < world.size(); i++) {
      size_t rows_for_proc = base_rows_per_proc + (i < remainder_rows ? 1 : 0);
      sendcounts[i] = static_cast<int>(rows_for_proc * cols);
      displs[i] = static_cast<int>(offset * cols);
      offset += rows_for_proc;
    }
  }

  loc_matrix_.resize(local_rows * cols);

  int* sendbuf = nullptr;
  if (world.rank() == 0) {
    sendbuf = reinterpret_cast<int*>(taskData->inputs[0]);
  }

  MPI_Scatterv(sendbuf, sendcounts.data(), displs.data(), MPI_INT, loc_matrix_.data(),
               static_cast<int>(loc_matrix_.size()), MPI_INT, 0, MPI_COMM_WORLD);

  loc_row_sums_.resize(local_rows, 0);

  for (size_t i = 0; i < local_rows; i++) {
    loc_row_sums_[i] = 0;
    for (size_t j = 0; j < cols; j++) {
      loc_row_sums_[i] += loc_matrix_[(i * cols) + j];
    }
  }

  if (world.rank() == 0) {
    row_sums_.resize(rows, 0);
  }

  sendcounts.clear();

  size_t offset = 0;
  for (int i = 0; i < world.size(); i++) {
    size_t rows_for_proc = base_rows_per_proc + (i < remainder_rows ? 1 : 0);
    sendcounts[i] = static_cast<int>(rows_for_proc);
    displs[i] = static_cast<int>(offset);
    offset += rows_for_proc;
  }

  MPI_Gatherv(loc_row_sums_.data(), static_cast<int>(loc_row_sums_.size()), MPI_INT, row_sums_.data(),
              sendcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool borisov_s_sum_of_rows::SumOfRowsTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    if (!row_sums_.empty()) {
      int* out = reinterpret_cast<int*>(taskData->outputs[0]);
      for (size_t i = 0; i < row_sums_.size(); i++) {
        out[i] = row_sums_[i];
      }
    }
  }
  return true;
}
