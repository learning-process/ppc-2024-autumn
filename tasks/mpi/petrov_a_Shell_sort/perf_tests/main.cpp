#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <vector>

#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

TEST(petrov_a_Shell_sort_mpi, test_task_run_mpi) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = 10000;
  int cols = 50;

  std::vector<int> global_matrix;
  std::vector<int> global_vector(cols, 1);
  std::vector<int> local_result;
  std::vector<int> global_result;

  if (rank == 0) {
    global_matrix.resize(rows * cols);
    for (size_t i = 0; i < global_matrix.size(); ++i) {
      global_matrix[i] = i + 1;
    }
  }

  MPI_Bcast(global_vector.data(), global_vector.size(), MPI_INT, 0, MPI_COMM_WORLD);

  int rows_per_proc = rows / size;
  int remainder = rows % size;
  int start_row = rank * rows_per_proc + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + static_cast<int>(rank < remainder);
  end_row = std::min(end_row, rows);

  std::vector<int> local_matrix((end_row - start_row) * cols);

  int* sendcounts = new int[size];
  int* displs = new int[size];
  for (int i = 0; i < size; ++i) {
    sendcounts[i] = (rows_per_proc + static_cast<int>(i < remainder)) * cols;
    displs[i] = (rows_per_proc * i + std::min(i, remainder)) * cols;
  }
  MPI_Scatterv(global_matrix.data(), sendcounts, displs, MPI_INT, local_matrix.data(), (end_row - start_row) * cols,
               MPI_INT, 0, MPI_COMM_WORLD);
  delete[] sendcounts;
  delete[] displs;

  local_result.resize(end_row - start_row);
  for (int i = 0; i < end_row - start_row; ++i) {
    local_result[i] = 0;
    for (int j = 0; j < cols; ++j) {
      local_result[i] += local_matrix[i * cols + j] * global_vector[j];
    }
  }

  if (rank == 0) {
    global_result.resize(rows);
    for (int i = 0; i < rows; ++i) {
      global_result[i] = 0;
      for (int j = 0; j < cols; ++j) {
        global_result[i] += (i * cols + j + 1) * global_vector[j];
      }
    }
  }

  sendcounts = new int[size];
  displs = new int[size];
  for (int i = 0; i < size; ++i) {
    sendcounts[i] = rows_per_proc + static_cast<int>(i < remainder);
    displs[i] = (rows_per_proc * i + std::min(i, remainder));
  }
  MPI_Gatherv(local_result.data(), local_result.size(), MPI_INT, global_result.data(), sendcounts, displs, MPI_INT, 0,
              MPI_COMM_WORLD);
  delete[] sendcounts;
  delete[] displs;

  if (rank == 0) {
    std::vector<int> expected_result = global_result;
    std::sort(expected_result.begin(), expected_result.end());

    EXPECT_EQ(global_result, expected_result);
  }
}

TEST(petrov_a_Shell_sort_mpi, test_pipeline_run_mpi) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = 2000;
  int cols = 100;

  std::vector<int> global_matrix;
  std::vector<int> global_vector(cols, 1);
  std::vector<int> local_result;
  std::vector<int> global_result;

  if (rank == 0) {
    global_matrix.resize(rows * cols);
    for (size_t i = 0; i < global_matrix.size(); ++i) {
      global_matrix[i] = i + 1;
    }
  }

  MPI_Bcast(global_vector.data(), global_vector.size(), MPI_INT, 0, MPI_COMM_WORLD);

  int rows_per_proc = rows / size;
  int remainder = rows % size;
  int start_row = rank * rows_per_proc + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + static_cast<int>(rank < remainder);
  end_row = std::min(end_row, rows);

  std::vector<int> local_matrix((end_row - start_row) * cols);

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  for (int i = 0; i < size; ++i) {
    sendcounts[i] = (rows_per_proc + static_cast<int>(i < remainder)) * cols;
    displs[i] = (rows_per_proc * i + std::min(i, remainder)) * cols;
  }

  MPI_Scatterv(global_matrix.data(), sendcounts.data(), displs.data(), MPI_INT, local_matrix.data(),
               (end_row - start_row) * cols, MPI_INT, 0, MPI_COMM_WORLD);

  local_result.resize(end_row - start_row);
  for (int i = 0; i < end_row - start_row; ++i) {
    local_result[i] = 0;
    for (int j = 0; j < cols; ++j) {
      local_result[i] += local_matrix[i * cols + j] * global_vector[j];
    }
  }

  if (rank == 0) {
    global_result.resize(rows);
    for (int i = 0; i < rows; ++i) {
      global_result[i] = 0;
      for (int j = 0; j < cols; ++j) {
        global_result[i] += (i * cols + j + 1) * global_vector[j];
      }
    }
  }

  std::vector<int> recvcounts(size);
  std::vector<int> recvdispls(size);
  for (int i = 0; i < size; ++i) {
    recvcounts[i] = rows_per_proc + static_cast<int>(i < remainder);
    recvdispls[i] = (rows_per_proc * i + std::min(i, remainder));
  }

  MPI_Gatherv(local_result.data(), local_result.size(), MPI_INT, global_result.data(), recvcounts.data(),
              recvdispls.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<int> expected_result = global_result;
    std::sort(expected_result.begin(), expected_result.end());

    EXPECT_EQ(global_result, expected_result);
  }
}
