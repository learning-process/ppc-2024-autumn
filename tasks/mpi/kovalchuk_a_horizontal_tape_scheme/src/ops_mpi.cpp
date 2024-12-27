#include "mpi/kovalchuk_a_horizontal_tape_scheme/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <random>
#include <string>
#include <vector>

bool kovalchuk_a_horizontal_tape_scheme::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init matrix and vector
  if (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0) {
    matrix_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], matrix_[i].begin());
    }
    vector_ = std::vector<int>(taskData->inputs_count[1]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[taskData->inputs_count[0]]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], vector_.begin());
  } else {
    matrix_ = std::vector<std::vector<int>>();
    vector_ = std::vector<int>();
  }
  // Init result vector
  result_ = std::vector<int>(taskData->inputs_count[0], 0);
  return true;
}

bool kovalchuk_a_horizontal_tape_scheme::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool kovalchuk_a_horizontal_tape_scheme::TestMPITaskSequential::run() {
  internal_order_test();
  if (!matrix_.empty() && !vector_.empty()) {
    for (unsigned int i = 0; i < matrix_.size(); i++) {
      result_[i] = std::inner_product(matrix_[i].begin(), matrix_[i].end(), vector_.begin(), 0);
    }
  }
  return true;
}

bool kovalchuk_a_horizontal_tape_scheme::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::copy(result_.begin(), result_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

bool kovalchuk_a_horizontal_tape_scheme::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  // Init matrix and vector to root
  if (world.rank() == 0) {
    int rows = taskData->inputs_count[0];
    int columns = taskData->inputs_count[1];

    if (rows > 0 && columns > 0) {
      matrix_ = std::vector<std::vector<int>>(rows, std::vector<int>(columns));
      for (int i = 0; i < rows; i++) {
        auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
        std::copy(tmp_ptr, tmp_ptr + columns, matrix_[i].begin());
      }
      vector_ = std::vector<int>(columns);
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[rows]);
      std::copy(tmp_ptr, tmp_ptr + columns, vector_.begin());
    } else {
      matrix_ = std::vector<std::vector<int>>();
      vector_ = std::vector<int>();
    }
  }

  return true;
}

bool kovalchuk_a_horizontal_tape_scheme::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == taskData->inputs_count[0];
  }
  return true;
}

bool kovalchuk_a_horizontal_tape_scheme::TestMPITaskParallel::run() {
  internal_order_test();

  int rank = world.rank();
  int size = world.size();

  int rows;
  int columns;
  if (world.rank() == 0) {
    rows = taskData->inputs_count[0];
    columns = taskData->inputs_count[1];
  }

  boost::mpi::broadcast(world, columns, 0);
  boost::mpi::broadcast(world, rows, 0);
  boost::mpi::broadcast(world, vector_, 0);

  int local_rows = rows / size;
  int extra_rows = rows % size;

  std::vector<int> sendcounts(size, local_rows * columns);
  for (int i = 0; i < extra_rows; ++i) {
    sendcounts[i] += columns;
  }

  std::vector<int> displs(size, 0);
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + sendcounts[i - 1];
  }

  std::vector<int> flattened_matrix;
  if (rank == 0) {
    // Flatten the matrix
    flattened_matrix.reserve(rows * columns);
    for (const auto& row : matrix_) {
      flattened_matrix.insert(flattened_matrix.end(), row.begin(), row.end());
    }
  }

  // Allocate space
  std::vector<int> local_matrix(sendcounts[rank]);

  // Scatter the matrix data
  boost::mpi::scatterv(world, flattened_matrix.data(), sendcounts, displs, local_matrix.data(), sendcounts[rank], 0);

  // Reshape local matrix
  int actual_local_rows = local_rows + (rank < extra_rows ? 1 : 0);
  matrix_ = std::vector<std::vector<int>>(actual_local_rows, std::vector<int>(columns));
  for (int i = 0; i < actual_local_rows; ++i) {
    std::copy(local_matrix.begin() + i * columns, local_matrix.begin() + (i + 1) * columns, matrix_[i].begin());
  }

  // Initialize local and global result vectors
  local_result_ = std::vector<int>(matrix_.size(), 0);
  result_ = std::vector<int>(rows, 0);

  if (!matrix_.empty() && !vector_.empty()) {
    for (unsigned int i = 0; i < matrix_.size(); i++) {
      local_result_[i] = std::inner_product(matrix_[i].begin(), matrix_[i].end(), vector_.begin(), 0);
    }
  }

  // Gather results
  std::vector<int> recvcounts(world.size(), rows / world.size());
  for (int i = 0; i < extra_rows; ++i) {
    recvcounts[i] += 1;
  }

  std::vector<int> recvdispls(world.size(), 0);
  for (int i = 1; i < world.size(); ++i) {
    recvdispls[i] = recvdispls[i - 1] + recvcounts[i - 1];
  }

  boost::mpi::gatherv(world, local_result_.data(), local_result_.size(), result_.data(), recvcounts, recvdispls, 0);
  return true;
}

bool kovalchuk_a_horizontal_tape_scheme::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(result_.begin(), result_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
