// Golovkin Maksim Task#2

#include "mpi/golovkin_rowwise_matrix_partitioning/include/ops_mpi.hpp"

#include <boost/mpi.hpp>

bool golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::validation() {
  internal_order_test();

  bool is_valid = true;
  if (world.size() < 5 || world.rank() >= 4) {
    rows_A = taskData->inputs_count[0];
    cols_A = taskData->inputs_count[1];
    rows_B = taskData->inputs_count[2];
    cols_B = taskData->inputs_count[3];
    if (cols_A != rows_B || rows_A <= 0 || cols_A <= 0 || rows_B <= 0 || cols_B <= 0) {
      is_valid = false;
    }
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::pre_processing() {
  internal_order_test();
  if (world.size() < 5 || world.rank() >= 4) {
    A = new double[rows_A * cols_A];
    B = new double[rows_B * cols_B];
    auto* tmp_A = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* tmp_B = reinterpret_cast<double*>(taskData->inputs[1]);
    std::copy(tmp_A, tmp_A + rows_A * cols_A, A);
    std::copy(tmp_B, tmp_B + rows_B * cols_B, B);
    result = nullptr;
  }
  return true;
}

bool golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::run() {
  internal_order_test();
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int dimensions[4];
  if (rank == 0) {
    dimensions[0] = rows_A;
    dimensions[1] = cols_A;
    dimensions[2] = rows_B;
    dimensions[3] = cols_B;
  }
  MPI_Bcast(dimensions, 4, MPI_INT, 0, MPI_COMM_WORLD);

  rows_A = dimensions[0];
  cols_A = dimensions[1];
  rows_B = dimensions[2];
  cols_B = dimensions[3];

  if (rank != 0) {
    A = new double[rows_A * cols_A];
    B = new double[rows_B * cols_B];
  }
  MPI_Bcast(A, rows_A * cols_A, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(B, rows_B * cols_B, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  int* counter_send = new int[size];
  std::fill(counter_send, counter_send + size, 0);
  int* displs = new int[size]();
  int rows_per_proc = rows_A / size;
  int rows_size = rows_A % size;
  int set_zero = 0;
  for (int i = 0; i < size; ++i) {
    if (i < rows_size) {
      counter_send[i] = (rows_per_proc + 1) * cols_A;
    } else {
      counter_send[i] = rows_per_proc * cols_A;
    }
    displs[i] = set_zero;
    set_zero += counter_send[i];
  }
  auto* local_A = new double[counter_send[rank]]();
  MPI_Scatterv(A, counter_send, displs, MPI_DOUBLE, local_A, counter_send[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int local_rows = counter_send[rank] / cols_A;
  auto* local_res = new double[local_rows * cols_B];
  std::fill(local_res, local_res + local_rows * cols_B, 0.0);
  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < cols_B; j++) {
      for (int k = 0; k < cols_A; k++) {
        local_res[i * cols_B + j] += local_A[i * cols_A + k] * B[k * cols_B + j];
      }
    }
  }

  if (rank == 0) {
    result = new double[rows_A * cols_B];
  } else {
    result = nullptr;
  }
  int* recvcounts = new int[size]();
  int* recvdispls = new int[size]();
  set_zero = 0;

  for (int i = 0; i < size; ++i) {
    if (i < rows_size) {
      recvcounts[i] = (rows_per_proc + 1) * cols_B;
    } else {
      recvcounts[i] = rows_per_proc * cols_B;
    }
    recvdispls[i] = set_zero;
    set_zero += recvcounts[i];
  };
  MPI_Gatherv(local_res, local_rows * cols_B, MPI_DOUBLE, result, recvcounts, recvdispls, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
  delete[] counter_send;
  delete[] displs;
  delete[] local_A;
  delete[] local_res;
  delete[] recvcounts;
  delete[] recvdispls;

  return true;
}

bool golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask::post_processing() {
  internal_order_test();
  if (world.size() < 5 || world.rank() >= 4) {
    std::memcpy(reinterpret_cast<double*>(taskData->outputs[0]), result, rows_A * cols_B * sizeof(double));
  }
  return true;
}