#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "mpi/petrov_a_ribbon_vertical_scheme/include/ops_mpi.hpp"

using namespace petrov_a_ribbon_vertical_scheme_mpi;

TEST(petrov_a_ribbon_vertical_scheme_mpi, HandlesSmallMatrix) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> matrix;
  std::vector<int> vector;
  std::vector<int> local_result;
  std::vector<int> result;

  if (rank == 0) {
    matrix = {1, 2, 3, 4, 5, 6};
    vector = {1, 1, 1};
  }

  int rows = 2;
  int cols = 3;
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    matrix.resize(rows * cols);
    vector.resize(cols);
  }

  MPI_Bcast(matrix.data(), matrix.size(), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(vector.data(), vector.size(), MPI_INT, 0, MPI_COMM_WORLD);

  int rows_per_proc = rows / size;
  int remainder = rows % size;
  int start_row = rank * rows_per_proc + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);

  local_result.resize(end_row - start_row);

  for (int i = start_row; i < end_row; ++i) {
    local_result[i - start_row] = 0;
    for (int j = 0; j < cols; ++j) {
      local_result[i - start_row] += matrix[i * cols + j] * vector[j];
    }
  }

  if (rank == 0) {
    result.resize(rows);
  }

  int* recvcounts = new int[size];
  int* displs = new int[size];

  for (int i = 0; i < size; i++) {
    recvcounts[i] = (rows / size) + (i < remainder ? 1 : 0);
    displs[i] = (rows / size) * i + std::min(i, remainder);
  }

  MPI_Gatherv(local_result.data(), end_row - start_row, MPI_INT, result.data(), recvcounts, displs, MPI_INT, 0,
              MPI_COMM_WORLD);

  delete[] recvcounts;
  delete[] displs;

  if (rank == 0) {
    ASSERT_EQ(result[0], 6);
    ASSERT_EQ(result[1], 15);
  }
}

TEST(petrov_a_ribbon_vertical_scheme_mpi, Handles3x2Matrix) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> matrix;
  std::vector<int> vector;
  std::vector<int> local_result;
  std::vector<int> result;

  if (rank == 0) {
    matrix = {1, 2, 3, 4, 5, 6};
    vector = {1, 2};
    result.resize(3);
  }

  int rows = 3;
  int cols = 2;
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    matrix.resize(rows * cols);
    vector.resize(cols);
  }

  MPI_Bcast(matrix.data(), matrix.size() * sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(vector.data(), vector.size() * sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);

  int rows_per_proc = rows / size;
  int remainder = rows % size;
  int start_row = rank * rows_per_proc + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
  local_result.resize(end_row - start_row);

  for (int i = start_row; i < end_row; ++i) {
    local_result[i - start_row] = 0;
    for (int j = 0; j < cols; ++j) {
      local_result[i - start_row] += matrix[i * cols + j] * vector[j];
    }
  }

  if (rank == 0) {
    result.resize(rows);
  }

  int* recvcounts = new int[size];
  int* displs = new int[size];

  for (int i = 0; i < size; ++i) {
    recvcounts[i] = (rows / size) + (i < remainder ? 1 : 0);
    displs[i] = (rows / size) * i + std::min(i, remainder);
  }

  MPI_Gatherv(local_result.data(), end_row - start_row, MPI_INT, result.data(), recvcounts, displs, MPI_INT, 0,
              MPI_COMM_WORLD);

  delete[] recvcounts;
  delete[] displs;

  if (rank == 0) {
    std::vector<int> expected = {5, 11, 17};
    ASSERT_EQ(result, expected);
  }
}

TEST(petrov_a_ribbon_vertical_scheme_mpi, Handles4x4Matrix) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> matrix;
  std::vector<int> vector;
  std::vector<int> local_result;
  std::vector<int> result;

  if (rank == 0) {
    matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    vector = {1, 2, 3, 4};
    result.resize(4);
  }

  int rows = 4;
  int cols = 4;
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    matrix.resize(rows * cols);
    vector.resize(cols);
  }

  MPI_Bcast(matrix.data(), matrix.size() * sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(vector.data(), vector.size() * sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);

  int rows_per_proc = rows / size;
  int remainder = rows % size;
  int start_row = rank * rows_per_proc + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
  local_result.resize(end_row - start_row);

  for (int i = start_row; i < end_row; ++i) {
    local_result[i - start_row] = 0;
    for (int j = 0; j < cols; ++j) {
      local_result[i - start_row] += matrix[i * cols + j] * vector[j];
    }
  }

  if (rank == 0) {
    result.resize(rows);
  }

  int* recvcounts = new int[size];
  int* displs = new int[size];

  for (int i = 0; i < size; ++i) {
    recvcounts[i] = (rows / size) + (i < remainder ? 1 : 0);
    displs[i] = (rows / size) * i + std::min(i, remainder);
  }

  MPI_Gatherv(local_result.data(), end_row - start_row, MPI_INT, result.data(), recvcounts, displs, MPI_INT, 0,
              MPI_COMM_WORLD);

  delete[] recvcounts;
  delete[] displs;

  if (rank == 0) {
    std::vector<int> expected = {30, 70, 110, 150};
    ASSERT_EQ(result, expected);
  }
}
TEST(petrov_a_ribbon_vertical_scheme_mpi, Handles5x3Matrix) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> matrix;
  std::vector<int> vector;
  std::vector<int> local_result;
  std::vector<int> result;

  if (rank == 0) {
    matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    vector = {1, 2, 3};
    result.resize(5);
  }

  int rows = 5;
  int cols = 3;
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    matrix.resize(rows * cols);
    vector.resize(cols);
  }

  MPI_Bcast(matrix.data(), rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(vector.data(), cols, MPI_INT, 0, MPI_COMM_WORLD);

  int rows_per_proc = rows / size;
  int remainder = rows % size;
  int start_row = rank * rows_per_proc + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + static_cast<int>(rank < remainder);
  end_row = std::min(end_row, rows);

  local_result.resize(end_row - start_row);
  for (int i = start_row; i < end_row; ++i) {
    local_result[i - start_row] = 0;
    for (int j = 0; j < cols; ++j) {
      local_result[i - start_row] += matrix[i * cols + j] * vector[j];
    }
  }

  if (rank == 0) {
    result.resize(rows);
  }

  int* recvcounts = new int[size];
  int* displs = new int[size];
  for (int i = 0; i < size; ++i) {
    recvcounts[i] = (rows / size) + (i < remainder ? 1 : 0);
    displs[i] = (rows / size) * i + std::min(i, remainder);
  }

  MPI_Gatherv(local_result.data(), end_row - start_row, MPI_INT, result.data(), recvcounts, displs, MPI_INT, 0,
              MPI_COMM_WORLD);
  delete[] recvcounts;
  delete[] displs;

  if (rank == 0) {
    std::vector<int> expected = {14, 32, 50, 68, 86};
    ASSERT_EQ(result, expected);
  }
}
TEST(petrov_a_ribbon_vertical_scheme_mpi, HandlesNonSquareMatrix) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> matrix;
  std::vector<int> vector;
  std::vector<int> local_result;
  std::vector<int> result;

  if (rank == 0) {
    matrix = {1, 2, 3, 4, 5, 6, 7, 8};
    vector = {1, 2};
    result.resize(4);
  }

  int rows = 4;
  int cols = 2;
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    matrix.resize(rows * cols);
    vector.resize(cols);
  }

  MPI_Bcast(matrix.data(), rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(vector.data(), cols, MPI_INT, 0, MPI_COMM_WORLD);

  int rows_per_proc = rows / size;
  int remainder = rows % size;
  int start_row = rank * rows_per_proc + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + static_cast<int>(rank < remainder);
  end_row = std::min(end_row, rows);

  local_result.resize(end_row - start_row);
  for (int i = start_row; i < end_row; ++i) {
    local_result[i - start_row] = 0;
    for (int j = 0; j < cols; ++j) {
      local_result[i - start_row] += matrix[i * cols + j] * vector[j];
    }
  }

  if (rank == 0) {
    result.resize(rows);
  }

  int* recvcounts = new int[size];
  int* displs = new int[size];
  for (int i = 0; i < size; ++i) {
    recvcounts[i] = (rows / size) + (i < remainder ? 1 : 0);
    displs[i] = (rows / size) * i + std::min(i, remainder);
  }
  MPI_Gatherv(local_result.data(), end_row - start_row, MPI_INT, result.data(), recvcounts, displs, MPI_INT, 0,
              MPI_COMM_WORLD);
  delete[] recvcounts;
  delete[] displs;

  if (rank == 0) {
    std::vector<int> expected = {5, 11, 17, 23};
    ASSERT_EQ(result, expected);
  }
}