#include "mpi/kalinin_d_matrix_mult_hor_a_vert_b/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  input_A = reinterpret_cast<int*>(taskData->inputs[0]);
  input_B = reinterpret_cast<int*>(taskData->inputs[1]);

  rows_A = taskData->inputs_count[0];
  columns_A = taskData->inputs_count[1];
  rows_B = taskData->inputs_count[2];
  columns_B = taskData->inputs_count[3];

  if (columns_A != rows_B) {
    return false;
  }

  C.assign(rows_A * columns_B, 0);

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->inputs_count[2] > 0 &&
         taskData->inputs_count[3] > 0 && taskData->inputs_count[1] == taskData->inputs_count[2];
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  for (int i = 0; i < rows_A; ++i) {
    for (int j = 0; j < columns_B; ++j) {
      for (int k = 0; k < columns_A; ++k) {
        C[i * columns_B + j] += input_A[i * columns_A + k] * input_B[k * columns_B + j];
      }
    }
  }

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  size_t total_size = rows_A * columns_B;
  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);

  for (size_t i = 0; i < total_size; ++i) {
    output_data[i] = C[i];
  }

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    rows_A = taskData->inputs_count[0];
    columns_A = taskData->inputs_count[1];
    rows_B = taskData->inputs_count[2];
    columns_B = taskData->inputs_count[3];

    input_A = new int[columns_A * rows_A];
    input_B = new int[columns_B * rows_B];
    auto* tmp_ptr_a = reinterpret_cast<int*>(taskData->inputs[0]);
    auto* tmp_ptr_b = reinterpret_cast<int*>(taskData->inputs[1]);

    for (int i = 0; i < columns_A * rows_A; i++) {
      input_A[i] = tmp_ptr_a[i];
    }

    for (int i = 0; i < columns_B * rows_B; i++) {
      input_B[i] = tmp_ptr_b[i];
    }
    C = nullptr;
  }

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->inputs_count[2] > 0 &&
           taskData->inputs_count[3] > 0 && taskData->inputs_count[1] == taskData->inputs_count[2];
  }
  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int dimensions[4];
  if (rank == 0) {
    dimensions[0] = taskData->inputs_count[1];
    dimensions[1] = taskData->inputs_count[0];
    dimensions[2] = taskData->inputs_count[3];
    dimensions[3] = taskData->inputs_count[2];
  }

  MPI_Bcast(dimensions, 4, MPI_INT, 0, MPI_COMM_WORLD);

  column_A = dimensions[0];
  row_A = dimensions[1];
  column_B = dimensions[2];
  row_B = dimensions[3];

  if (column_A != row_B) {
    if (rank == 0) {
      std::cerr << "Matrix dimensions are incompatible for multiplication: "
                << "A(" << row_A << "x" << column_A << "), "
                << "B(" << row_B << "x" << column_B << ")." << std::endl;
    }
    return false;
  }

  if (rank != 0) {
    input_A = new int[column_A * row_A]();
    input_B = new int[column_B * row_B]();
  }

  if (input_A == nullptr || input_B == nullptr) {
    std::cerr << "Error: input_A or input_B is a null pointer!" << std::endl;
    return false;
  }

  MPI_Bcast(input_A, column_A * row_A, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(input_B, column_B * row_B, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  int* sendcounts = new int[size];
  int* displs = new int[size];
  std::fill(sendcounts, sendcounts + size, 0);
  std::fill(displs, displs + size, 0);

  int rows_per_proc = row_A / size;
  int extra_rows = row_A % size;
  int offset = 0;

  for (int i = 0; i < size; ++i) {
    sendcounts[i] = (rows_per_proc + (i < extra_rows ? 1 : 0)) * column_A;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  auto* local_A = new int[sendcounts[rank]]();
  MPI_Scatterv(input_A, sendcounts, displs, MPI_INT, local_A, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  int local_rows = sendcounts[rank] / column_A;
  auto* local_res = new int[local_rows * column_B];
  std::fill(local_res, local_res + local_rows * column_B, 0);

  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < column_B; ++j) {
      for (int k = 0; k < column_A; ++k) {
        local_res[i * column_B + j] += local_A[i * column_A + k] * input_B[k * column_B + j];
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    C = new int[row_A * column_B];
  }

  int* recvcounts = new int[size];
  int* recvdispls = new int[size];
  offset = 0;

  for (int i = 0; i < size; ++i) {
    recvcounts[i] = (rows_per_proc + (i < extra_rows ? 1 : 0)) * column_B;
    recvdispls[i] = offset;
    offset += recvcounts[i];
  }

  MPI_Gatherv(local_res, local_rows * column_B, MPI_INT, C, recvcounts, recvdispls, MPI_INT, 0, MPI_COMM_WORLD);

  delete[] sendcounts;
  delete[] displs;
  delete[] local_A;
  delete[] local_res;
  delete[] recvcounts;
  delete[] recvdispls;

  return true;
}

bool kalinin_d_matrix_mult_hor_a_vert_b_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    size_t total_size = rows_A * columns_B;

    int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);

    for (size_t i = 0; i < total_size; ++i) {
      output_data[i] = C[i];
    }
  }

  return true;
}
