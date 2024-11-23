// Copyright 2023 Nesterov Alexander
#include "mpi/smirnov_i_tape_splitting_A/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool smirnov_i_tape_splitting_A::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  m_a = taskData->inputs_count[0];
  n_a = taskData->inputs_count[1];
  m_b = taskData->inputs_count[2];
  n_b = taskData->inputs_count[3];
  A = new double[m_a * n_a];
  B = new double[m_b * n_b];
  auto* tmp_ptr_a = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* tmp_ptr_b = reinterpret_cast<double*>(taskData->inputs[1]);
  for (int i = 0; i < m_a * n_a; i++) {
    A[i] = tmp_ptr_a[i];
  }
  for (int i = 0; i < m_b * n_b; i++) {
    B[i] = tmp_ptr_b[i];
  }

  // Init value for output
  res = nullptr;
  return true;
}

bool smirnov_i_tape_splitting_A::TestMPITaskSequential::validation() {
  internal_order_test();
  m_a = taskData->inputs_count[0];
  n_a = taskData->inputs_count[1];
  m_b = taskData->inputs_count[2];
  n_b = taskData->inputs_count[3];
  if (n_a != m_b || m_a <= 0 || n_a <= 0 || m_b <= 0 || n_b <= 0) {
    return false;
  }
  return true;
}

bool smirnov_i_tape_splitting_A::TestMPITaskSequential::run() {
  internal_order_test();
  res = new double[m_a * n_b];
  for (int i = 0; i < m_a * n_b; i++) {
    res[i] = 0;
  }
  for (int i = 0; i < m_a; i++) {
    for (int j = 0; j < n_b; j++) {
      for (int k = 0; k < n_a; k++) {
        res[i * n_b + j] += A[i * n_a + k] * B[k * n_b + j];
      }
    }
  }
  return true;
}

bool smirnov_i_tape_splitting_A::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  for (int i = 0; i < m_a * n_b; i++) {
    tmp_ptr[i] = res[i];
  }
  return true;
}

bool smirnov_i_tape_splitting_A::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // Init vectors
    m_a = taskData->inputs_count[0];
    n_a = taskData->inputs_count[1];
    m_b = taskData->inputs_count[2];
    n_b = taskData->inputs_count[3];
    A = new double[m_a * n_a];
    B = new double[m_b * n_b];
    auto* tmp_ptr_a = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* tmp_ptr_b = reinterpret_cast<double*>(taskData->inputs[1]);
    for (int i = 0; i < m_a * n_a; i++) {
      A[i] = tmp_ptr_a[i];
    }
    for (int i = 0; i < m_b * n_b; i++) {
      B[i] = tmp_ptr_b[i];
    }

    res = nullptr;
  }
  return true;
}

bool smirnov_i_tape_splitting_A::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    m_a = taskData->inputs_count[0];
    n_a = taskData->inputs_count[1];
    m_b = taskData->inputs_count[2];
    n_b = taskData->inputs_count[3];
    if (n_a != m_b || m_a <= 0 || n_a <= 0 || m_b <= 0 || n_b <= 0) {
      return false;
    }
  }
  return true;
}

bool smirnov_i_tape_splitting_A::TestMPITaskParallel::run() {
  internal_order_test();
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int dims[4];
  if (rank == 0) {
    dims[0] = m_a;
    dims[1] = n_a;
    dims[2] = m_b;
    dims[3] = n_b;
  }
  MPI_Bcast(dims, 4, MPI_INT, 0, MPI_COMM_WORLD);

  m_a = dims[0];
  n_a = dims[1];
  m_b = dims[2];
  n_b = dims[3];

  if (rank != 0) {
    A = new double[m_a * n_a];
    B = new double[m_b * n_b];
  }
  MPI_Bcast(A, m_a * n_a, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(B, m_b * n_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int* sendcounts = new int[size];
  int* displs = new int[size];
  int rows_per_proc = m_a / size;
  int extra_rows = m_a % size;
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    if (i < extra_rows) {
      sendcounts[i] = (rows_per_proc + 1) * n_a;
    } else {
      sendcounts[i] = rows_per_proc * n_a;
    }
    displs[i] = offset;
    offset += sendcounts[i];
  }
  double* local_A = new double[sendcounts[rank]];
  MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, local_A, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int local_rows = sendcounts[rank] / n_a;
  double* local_res = new double[local_rows * n_b]();

  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < n_b; j++) {
      for (int k = 0; k < n_a; k++) {
        local_res[i * n_b + j] += local_A[i * n_a + k] * B[k * n_b + j];
      }
    }
  }

  if (rank == 0) {
    res = new double[m_a * n_b];
  } else {
    res = nullptr;
  }
  int* recvcounts = new int[size];
  int* recvdispls = new int[size];
  offset = 0;

  for (int i = 0; i < size; ++i) {
    if (i < extra_rows) {
      recvcounts[i] = (rows_per_proc + 1) * n_b;
    } else {
      recvcounts[i] = rows_per_proc * n_b;
    }
    recvdispls[i] = offset;
    offset += recvcounts[i];
  };
  MPI_Gatherv(local_res, local_rows * n_b, MPI_DOUBLE, res, recvcounts, recvdispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  delete[] sendcounts;
  delete[] displs;
  delete[] local_A;
  delete[] local_res;
  delete[] recvcounts;
  delete[] recvdispls;

  return true;
}

bool smirnov_i_tape_splitting_A::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < m_a * n_b; i++) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}
