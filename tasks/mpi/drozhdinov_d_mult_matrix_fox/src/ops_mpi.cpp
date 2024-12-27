// Copyright 2023 Nesterov Alexander
#include "mpi/drozhdinov_d_mult_matrix_fox/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

using namespace std::chrono_literals;

void drozhdinov_d_mult_matrix_fox_mpi::SimpleMult(const std::vector<double>& A, const std::vector<double>& B,
                                                  std::vector<double>& C, int block) {
  for (int i = 0; i < block; ++i) {
    for (int j = 0; j < block; ++j) {
      for (int k = 0; k < block; ++k) {
        C[i * block + j] += A[i * block + k] * B[k * block + j];
      }
    }
  }
}

std::vector<double> drozhdinov_d_mult_matrix_fox_mpi::paddingMatrix(const std::vector<double>& mat, int rows, int cols,
                                                                    int padding) {
  std::vector<double> padded(padding * padding, 0.0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      padded[i * padding + j] = mat[i * cols + j];
    }
  }
  return padded;
}

std::vector<double> drozhdinov_d_mult_matrix_fox_mpi::SequentialFox(const std::vector<double>& A,
                                                                    const std::vector<double>& B, int k, int l, int n) {
  int C_rows = k;
  int C_cols = n;
  int padding = 1;
  while (padding < std::max({k, l, n})) {
    padding *= 2;
  }

  auto squareA = paddingMatrix(A, k, l, padding);
  auto squareB = paddingMatrix(B, l, n, padding);
  std::vector<double> squareC(padding * padding, 0.0);
  int grid_size = padding;
  int block_size = 1;

  std::vector<double> block_A(block_size * block_size);
  std::vector<double> block_B(block_size * block_size);
  std::vector<double> block_C(block_size * block_size, 0.0);

  for (int step = 0; step < grid_size; ++step) {
    for (int row = 0; row < grid_size; ++row) {
      for (int col = 0; col < grid_size; ++col) {
        int pivot = (row + step) % grid_size;

        for (int i = 0; i < block_size; ++i) {
          for (int j = 0; j < block_size; ++j) {
            block_A[i * block_size + j] = squareA[(row * block_size + i) * padding + (pivot * block_size + j)];
            block_B[i * block_size + j] = squareB[(pivot * block_size + i) * padding + (col * block_size + j)];
          }
        }

        SimpleMult(block_A, block_B, block_C, block_size);

        for (int i = 0; i < block_size; ++i) {
          for (int j = 0; j < block_size; ++j) {
            squareC[(row * block_size + i) * padding + (col * block_size + j)] += block_C[i * block_size + j];
          }
        }

        std::fill(block_C.begin(), block_C.end(), 0.0);
      }
    }
  }

  std::vector<double> C(C_rows * C_cols);
  for (int i = 0; i < C_rows; ++i) {
    for (int j = 0; j < C_cols; ++j) {
      C[i * C_cols + j] = squareC[i * padding + j];
    }
  }
  return C;
}

std::vector<double> drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel::ParallelFox(const std::vector<double>& a,
                                                                                       const std::vector<double>& b,
                                                                                       int K, int L, int N) {
  int size = K;
  std::vector<double> squareA;
  std::vector<double> squareB;

  int grid_size = static_cast<int>(sqrt(world.size()));

  MPI_Comm GCOMM;
  MPI_Comm CCOMM;
  MPI_Comm RCOMM;
  std::vector<int> grid_coords(2);
  std::vector<int> dim_size(2, grid_size);
  std::vector<int> periodic(2, 0);
  std::vector<int> subdims(2);
  MPI_Cart_create(MPI_COMM_WORLD, 2, dim_size.data(), periodic.data(), 0, &GCOMM);
  MPI_Cart_coords(GCOMM, world.rank(), 2, grid_coords.data());
  subdims[0] = 0;
  subdims[1] = 1;
  MPI_Cart_sub(GCOMM, subdims.data(), &RCOMM);
  subdims[0] = 1;
  subdims[1] = 0;
  MPI_Cart_sub(GCOMM, subdims.data(), &CCOMM);
  boost::mpi::communicator GRID_COMM(GCOMM, boost::mpi::comm_take_ownership);
  boost::mpi::communicator ROW_COMM(RCOMM, boost::mpi::comm_take_ownership);
  boost::mpi::communicator COL_COMM(CCOMM, boost::mpi::comm_take_ownership);
  int block_size;
  if (world.rank() == 0) {
    int padding = 1;
    while (padding < std::max({K, L, N})) {
      padding *= 2;
    }
    if (padding % grid_size != 0) {
      padding *= grid_size;
    }
    size = padding;
    block_size = padding / grid_size;
    squareA = paddingMatrix(a, K, L, padding);
    squareB = paddingMatrix(b, L, N, padding);
    std::vector<double> squareC(padding * padding, 0.0);
  }
  broadcast(world, block_size, 0);
  std::vector<double> block_A(block_size * block_size);
  std::vector<double> block_B(block_size * block_size);
  std::vector<double> block_AB(block_size * block_size, 0);
  if (world.rank() == 0) {
    for (int i = 0; i < block_size; i++) {
      for (int j = 0; j < block_size; j++) {
        block_A[i * block_size + j] = squareA[i * size + j];
        block_B[i * block_size + j] = squareB[i * size + j];
      }
    }
  }
  if (GRID_COMM.rank() == 0) {
    for (int p = 1; p < world.size(); p++) {
      int row = p / grid_size;
      int col = p % grid_size;
      std::vector<double> block_A_to_send(block_size * block_size);
      std::vector<double> block_B_to_send(block_size * block_size);

      for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
          block_A_to_send[i * block_size + j] = squareA[(row * block_size + i) * size + col * block_size + j];
          block_B_to_send[i * block_size + j] = squareB[(row * block_size + i) * size + col * block_size + j];
        }
      }
      GRID_COMM.send(p, 0, block_A_to_send);
      GRID_COMM.send(p, 1, block_B_to_send);
    }
  } else {
    GRID_COMM.recv(0, 0, block_A);
    GRID_COMM.recv(0, 1, block_B);
  }
  MPI_Status stat;
  for (int i = 0; i < grid_size; i++) {
    std::vector<double> tmpblockA(block_size * block_size);
    int pivot = (grid_coords[0] + i) % grid_size;
    if (grid_coords[1] == pivot) {
      tmpblockA = block_A;
    }
    broadcast(ROW_COMM, tmpblockA.data(), block_size * block_size, pivot);
    SimpleMult(tmpblockA, block_B, block_AB, block_size);
    int nextPr = grid_coords[0] + 1;
    if (grid_coords[0] == grid_size - 1) nextPr = 0;
    int prevPr = grid_coords[0] - 1;
    if (grid_coords[0] == 0) prevPr = grid_size - 1;
    MPI_Sendrecv_replace(block_B.data(), block_size * block_size, MPI_DOUBLE, prevPr, 0, nextPr, 0, COL_COMM, &stat);
  }
  std::vector<double> resultM(size * size);
  if (world.rank() == 0) {
    for (int i = 0; i < block_size; i++) {
      for (int j = 0; j < block_size; j++) {
        resultM[i * size + j] = block_AB[i * block_size + j];
      }
    }

    for (int p = 1; p < world.size(); p++) {
      int row = p / grid_size;
      int col = p % grid_size;

      std::vector<double> block_result(block_size * block_size);
      world.recv(p, 3, block_result);

      for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
          resultM[(row * block_size + i) * size + col * block_size + j] = block_result[i * block_size + j];
        }
      }
    }
  } else {
    world.send(0, 3, block_AB);
  }
  std::vector<double> final(K * N);
  if (world.rank() == 0) {
    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        final[i * N + j] = resultM[i * size + j];
      }
    }
  }
  return final;
}

bool drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  k = taskData->inputs_count[0];
  l = taskData->inputs_count[1];
  m = taskData->inputs_count[2];
  n = taskData->inputs_count[3];
  A.resize(k * l);
  B.resize(m * n);
  auto* ptra = reinterpret_cast<double*>(taskData->inputs[0]);
  for (int i = 0; i < k * l; i++) {
    A[i] = ptra[i];
  }
  auto* ptrb = reinterpret_cast<double*>(taskData->inputs[1]);
  for (int i = 0; i < m * n; i++) {
    B[i] = ptrb[i];
  }
  return true;
}

bool drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[1] == taskData->inputs_count[2] && taskData->inputs.size() == 2 &&
         taskData->outputs.size() == 1 && taskData->outputs_count[0] == taskData->inputs_count[0] &&
         taskData->outputs_count[1] == taskData->inputs_count[3];
}

bool drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  C = drozhdinov_d_mult_matrix_fox_mpi::SequentialFox(A, B, k, l, n);
  return true;
}

bool drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < k * n; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = C[i];
  }
  return true;
}

bool drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    k = taskData->inputs_count[0];
    l = taskData->inputs_count[1];
    m = taskData->inputs_count[2];
    n = taskData->inputs_count[3];
    A.resize(k * l);
    B.resize(m * n);
    auto* ptra = reinterpret_cast<double*>(taskData->inputs[0]);
    for (int i = 0; i < k * l; i++) {
      A[i] = ptra[i];
    }
    auto* ptrb = reinterpret_cast<double*>(taskData->inputs[1]);
    for (int i = 0; i < m * n; i++) {
      B[i] = ptrb[i];
    }
  }
  return true;
}

bool drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // int sq = static_cast<int>(std::sqrt(world.size()));
    // std::cout << sq << " " << world.size() << std::endl;
    // Check count elements of output
    return taskData->inputs_count[1] == taskData->inputs_count[2] && taskData->inputs.size() == 2 &&
           taskData->outputs.size() == 1 && taskData->outputs_count[0] == taskData->inputs_count[0] &&
           taskData->outputs_count[1] == taskData->inputs_count[3];  // &&(sq * sq == world.size())
  }
  return true;
}

bool drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  double sq = std::sqrt(world.size());
  if (sq != static_cast<int>(sq) || world.size() == 1) {
    C = drozhdinov_d_mult_matrix_fox_mpi::SequentialFox(A, B, k, l, n);
    return true;
  }
  C = drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel::ParallelFox(A, B, k, l, n);
  return true;
}

bool drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < k * n; i++) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = C[i];
    }
  }
  return true;
}
