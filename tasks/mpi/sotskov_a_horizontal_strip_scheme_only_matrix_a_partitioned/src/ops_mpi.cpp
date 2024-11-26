#include "mpi/sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  input_A = reinterpret_cast<int*>(taskData->inputs[0]);
  input_B = reinterpret_cast<int*>(taskData->inputs[1]);

  column_A = taskData->inputs_count[0];
  row_A = taskData->inputs_count[1];
  column_B = taskData->inputs_count[2];
  row_B = taskData->inputs_count[3];

  C.assign(column_A * row_B, 0);

  return true;
}

bool sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 1 && taskData->inputs_count[1] > 0 && taskData->inputs_count[2] > 0 &&
         taskData->inputs_count[3] > 0 && taskData->inputs_count[1] == taskData->inputs_count[2];
}

bool sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  int tmp;
  for (int i = 0; i < column_A; i++) {
    for (int j = 0; j < row_B; j++) {
      C[i * row_B + j] = 0;
      tmp = 0;
      for (int k = 0; k < row_A; k++) {
        tmp += input_A[i * row_A + k] * input_B[k * row_B + j];
        C[i * row_B + j] = tmp;
      }
    }
  }

  return true;
}

bool sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  int* ans = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(C.begin(), C.end(), ans);

  return true;
}

bool sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init vectors

    column_A = taskData->inputs_count[0];
    row_A = taskData->inputs_count[1];
    column_B = taskData->inputs_count[2];
    row_B = taskData->inputs_count[3];
    input_A = new int[column_A * row_A];
    input_B = new int[column_B * row_B];
    auto* tmp_ptr_a = reinterpret_cast<int*>(taskData->inputs[0]);
    auto* tmp_ptr_b = reinterpret_cast<int*>(taskData->inputs[1]);
    for (int i = 0; i < column_A * row_A; i++) {
      input_A[i] = tmp_ptr_a[i];
    }

    for (int i = 0; i < column_B * row_B; i++) {
      input_B[i] = tmp_ptr_b[i];
    }
    C = nullptr;
  }

  return true;
}
bool sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 1 && taskData->inputs_count[1] > 0 &&
           taskData->inputs_count[0] % taskData->inputs_count[1] == 0;
  }
  return true;
}

bool sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle::run() {
  internal_order_test();
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int d[4];
  if (rank == 0) {
    d[0] = column_A;
    d[1] = row_A;
    d[2] = column_B;
    d[3] = row_B;
  }
  MPI_Bcast(d, 4, MPI_INT, 0, MPI_COMM_WORLD);

  column_A = d[0];
  row_A = d[1];
  column_B = d[2];
  row_B = d[3];

  if (rank != 0) {
    input_A = new int[column_A * row_A];
    input_B = new int[column_B * row_B];
  }

  MPI_Bcast(input_A, column_A * row_A, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(input_B, column_B * row_B, MPI_INT, 0, MPI_COMM_WORLD);

  int* sendcounts = new int[size];
  std::fill(sendcounts, sendcounts + size, 0);
  int* displs = new int[size]();
  int rows_per_proc = column_A / size;
  int extra_rows = column_A % size;
  int offset = 0;

  for (int i = 0; i < size; ++i) {
    if (i < extra_rows) {
      sendcounts[i] = (rows_per_proc + 1) * row_A;
    } else {
      sendcounts[i] = rows_per_proc * row_A;
    }
    displs[i] = offset;
    offset += sendcounts[i];
  }

  auto* local_A = new int[sendcounts[rank]]();
  MPI_Scatterv(input_A, sendcounts, displs, MPI_INT, local_A, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  int local_rows = sendcounts[rank] / row_A;
  auto* local_res = new int[local_rows * row_B];
  std::fill(local_res, local_res + local_rows * row_B, 0);

  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < row_B; j++) {
      for (int k = 0; k < row_A; k++) {
        local_res[i * row_B + j] += local_A[i * row_A + k] * input_B[k * row_B + j];
      }
    }
  }

  if (rank == 0) {
    C = new int[column_A * row_B];
  } else {
    C = nullptr;
  }

  int* recvcounts = new int[size]();
  int* recvdispls = new int[size]();
  offset = 0;

  for (int i = 0; i < size; ++i) {
    if (i < extra_rows) {
      recvcounts[i] = (rows_per_proc + 1) * row_B;
    } else {
      recvcounts[i] = rows_per_proc * row_B;
    }
    recvdispls[i] = offset;
    offset += recvcounts[i];
  };

  MPI_Gatherv(local_res, local_rows * row_B, MPI_INT, C, recvcounts, recvdispls, MPI_INT, 0, MPI_COMM_WORLD);
  delete[] sendcounts;
  delete[] displs;
  delete[] local_A;
  delete[] local_res;
  delete[] recvcounts;
  delete[] recvdispls;

  return true;
}

bool sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int* ans = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(C, C + column_A * row_B, ans);
  }

  return true;
}
