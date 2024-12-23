#include "mpi/alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq::pre_processing() {
  internal_order_test();

  // //std::cout << "init A\n";
  double* input_A = reinterpret_cast<double*>(taskData->inputs[0]);
  N = static_cast<int>(taskData->inputs_count[0]);  // Размерность матриц (N x N)

  // //std::cout << "init B\n";
  double* input_B = reinterpret_cast<double*>(taskData->inputs[1]);

  A.resize(N * N);
  B.resize(N * N);

  for (int i = 0; i < N * N; ++i) {
    A[i] = input_A[i];
  }
  for (int i = 0; i < N * N; ++i) {
    B[i] = input_B[i];
  }
  C.resize(N * N, 0.0);

  return true;
}

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq::validation() {
  internal_order_test();

  return static_cast<int>(taskData->inputs_count[0]) > 1;
}

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq::run() {
  internal_order_test();

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        C[i * N + j] += A[i * N + k] * B[k * N + j];
      }
    }
  }

  return true;
}

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq::post_processing() {
  internal_order_test();

  double* res = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(C.begin(), C.end(), res);
  return true;
}

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_mpi::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    double* input_A = reinterpret_cast<double*>(taskData->inputs[0]);
    N = static_cast<int>(taskData->inputs_count[0]);
    double* input_B = reinterpret_cast<double*>(taskData->inputs[1]);

    A.resize(N * N);
    B.resize(N * N);

    for (int i = 0; i < N * N; ++i) {
      A[i] = input_A[i];
    }
    for (int i = 0; i < N * N; ++i) {
      B[i] = input_B[i];
    }

    C.resize(N * N, 0.0);
  }

  return true;
}

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_mpi::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return static_cast<int>(taskData->inputs_count[0]) > 1;
  }

  return true;
}

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_mpi::run() {
  internal_order_test();

  int total_procs, proc_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total_procs);

  int padded_size = N;  // N is assumed to be defined globally or elsewhere
  int grid_dimension = static_cast<int>(std::sqrt(total_procs));

  // Create 2D Cartesian communicator
  int grid_dims[2] = {grid_dimension, grid_dimension};
  int grid_periods_arr[2] = {0, 0};  // Non-periodic
  MPI_Comm grid_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, grid_dims, grid_periods_arr, 0, &grid_comm);

  // Get coordinates in the grid
  int coords_in_grid[2];
  MPI_Cart_coords(grid_comm, proc_rank, 2, coords_in_grid);

  // Create row and column communicators
  int row_dims[2] = {0, 1};  // Keep the second dimension (columns)
  MPI_Comm row_comm;
  MPI_Cart_sub(grid_comm, row_dims, &row_comm);

  int col_dims[2] = {1, 0};  // Keep the first dimension (rows)
  MPI_Comm col_comm;
  MPI_Cart_sub(grid_comm, col_dims, &col_comm);

  // Determine padding and block size
  int local_block_size;
  if (proc_rank == 0) {
    int padding = 1;
    while (padding < N) padding <<= 1;

    if (padding % grid_dimension != 0) {
      padding *= grid_dimension;
    }
    padded_size = padding;
    local_block_size = padding / grid_dimension;

    // Pad the matrices
    rectA.resize(padding * padding, 0.0);
    rectB.resize(padding * padding, 0.0);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        rectA[i * padding + j] = A[i * N + j];
        rectB[i * padding + j] = B[i * N + j];
      }
    }
  }

  // Broadcast block_size from root to all processes
  MPI_Bcast(&local_block_size, 1, MPI_INT, 0, grid_comm);

  // Broadcast the padded size as well, in case it's needed
  MPI_Bcast(&padded_size, 1, MPI_INT, 0, grid_comm);

  // Allocate memory for local blocks
  localA.resize(local_block_size * local_block_size, 0.0);
  localB.resize(local_block_size * local_block_size, 0.0);
  mult_block.resize(local_block_size * local_block_size, 0.0);

  std::vector<double> send_localA;
  std::vector<double> send_localB;

  if (proc_rank == 0) {
    // Initialize own localA and localB
    for (int i = 0; i < local_block_size; i++) {
      for (int j = 0; j < local_block_size; j++) {
        localA[i * local_block_size + j] = rectA[i * padded_size + j];
        localB[i * local_block_size + j] = rectB[i * padded_size + j];
      }
    }

    // Distribute blocks to other processes
    for (int proc_id = 1; proc_id < total_procs; proc_id++) {
      int proc_row = proc_id / grid_dimension;
      int proc_col = proc_id % grid_dimension;

      send_localA.resize(local_block_size * local_block_size);
      send_localB.resize(local_block_size * local_block_size);

      for (int i = 0; i < local_block_size; i++) {
        for (int j = 0; j < local_block_size; j++) {
          send_localA[i * local_block_size + j] =
              rectA[(proc_row * local_block_size + i) * padded_size + proc_col * local_block_size + j];
          send_localB[i * local_block_size + j] =
              rectB[(proc_row * local_block_size + i) * padded_size + proc_col * local_block_size + j];
        }
      }
      // Send localA with tag 0
      MPI_Send(send_localA.data(), local_block_size * local_block_size, MPI_DOUBLE, proc_id, 0, grid_comm);
      // Send localB with tag 1
      MPI_Send(send_localB.data(), local_block_size * local_block_size, MPI_DOUBLE, proc_id, 1, grid_comm);
    }
  } else {
    // Receive localA with tag 0
    MPI_Recv(localA.data(), local_block_size * local_block_size, MPI_DOUBLE, 0, 0, grid_comm, MPI_STATUS_IGNORE);
    // Receive localB with tag 1
    MPI_Recv(localB.data(), local_block_size * local_block_size, MPI_DOUBLE, 0, 1, grid_comm, MPI_STATUS_IGNORE);
  }

  // Perform the multiplication using Cannon's algorithm
  MPI_Status mpi_status;
  std::vector<double> broadcast_localA(local_block_size * local_block_size, 0.0);
  std::vector<double> recv_block_result(local_block_size * local_block_size, 0.0);
  std::vector<double> gathered_result;

  for (int i1 = 0; i1 < grid_dimension; i1++) {
    int broadcast_root = (coords_in_grid[0] + i1) % grid_dimension;

    if (coords_in_grid[1] == broadcast_root) {
      broadcast_localA = localA;
    }

    // Broadcast broadcast_localA within the current row
    MPI_Bcast(broadcast_localA.data(), local_block_size * local_block_size, MPI_DOUBLE, broadcast_root, row_comm);

    // Multiply the received block with localB and accumulate the result
    for (int i = 0; i < local_block_size; ++i) {
      for (int j = 0; j < local_block_size; ++j) {
        for (int k = 0; k < local_block_size; ++k) {
          mult_block[i * local_block_size + j] +=
              broadcast_localA[i * local_block_size + k] * localB[k * local_block_size + j];
        }
      }
    }

    // Shift localB upwards in the column communicator
    int nextPr = (coords_in_grid[0] + 1) % grid_dimension;
    int prevPr = (coords_in_grid[0] - 1 + grid_dimension) % grid_dimension;
    MPI_Sendrecv_replace(localB.data(), local_block_size * local_block_size, MPI_DOUBLE, prevPr, 0, nextPr, 0, col_comm,
                         &mpi_status);
  }

  // Gather the results to the root process
  if (proc_rank == 0) {
    gathered_result.resize(padded_size * padded_size, 0.0);
    // Copy own computed block to gathered_result
    for (int i = 0; i < local_block_size; i++) {
      for (int j = 0; j < local_block_size; j++) {
        gathered_result[i * padded_size + j] = mult_block[i * local_block_size + j];
      }
    }

    // Receive computed blocks from other processes
    for (int proc_id = 1; proc_id < total_procs; proc_id++) {
      int proc_row = proc_id / grid_dimension;
      int proc_col = proc_id % grid_dimension;
      recv_block_result.resize(local_block_size * local_block_size, 0.0);
      std::cout << proc_rank << " принял данные с " << proc_id << " прцесса" << std::endl;
      MPI_Recv(recv_block_result.data(), local_block_size * local_block_size, MPI_DOUBLE, proc_id, 3, grid_comm,
               MPI_STATUS_IGNORE);

      for (int i = 0; i < local_block_size; i++) {
        for (int j = 0; j < local_block_size; j++) {
          gathered_result[(proc_row * local_block_size + i) * padded_size + (proc_col * local_block_size + j)] =
              recv_block_result[i * local_block_size + j];
        }
      }
    }
  } else {
    std::cout << proc_rank << " отправил данные на 0 процесс" << std::endl;
    // Send computed mult_block to the root process with tag 3
    MPI_Send(mult_block.data(), local_block_size * local_block_size, MPI_DOUBLE, 0, 3, grid_comm);
  }

  // Extract the final result matrix (remove padding if necessary)
  std::vector<double> final_matrix;
  if (proc_rank == 0) {
    final_matrix.resize(N * N, 0.0);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        final_matrix[i * N + j] = gathered_result[i * padded_size + j];
      }
    }

    /*
    // Optionally, print the final matrix
    std::cout << "Final Matrix (N x N):\n";
    for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                    std::cout << final_matrix[i * N + j] << " ";
            }
            std::cout << "\n";
    }
    */
    C = final_matrix;
  }

  // Clean up communicators
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  MPI_Comm_free(&grid_comm);

  if (proc_rank != 0) {
    MPI_Comm_free(&grid_comm);
  }

  return true;
}

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_mpi::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    double* res = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(C.begin(), C.end(), res);
  }
  return true;
}