#include "mpi/alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_seq::pre_processing() {
  internal_order_test();

  // std::cout << "init A\n";
  auto* input_A = reinterpret_cast<double*>(taskData->inputs[0]);
  N = static_cast<int>(taskData->inputs_count[0]);  // Размерность матриц (N x N)

  // std::cout << "init B\n";
  auto* input_B = reinterpret_cast<double*>(taskData->inputs[1]);

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

  return static_cast<int>(taskData->inputs_count[0]) > 1;  // Проверка, что размерность матриц положительна
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

  auto* res = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(C.begin(), C.end(), res);
  return true;
}

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_mpi::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* input_A = reinterpret_cast<double*>(taskData->inputs[0]);
    N = static_cast<int>(taskData->inputs_count[0]);
    auto* input_B = reinterpret_cast<double*>(taskData->inputs[1]);

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

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int sqrtsize = static_cast<int>(sqrt(static_cast<double>(size)));

  std::vector<int> coordinates_of_grid(2);
  std::vector<int> dimSize(2, sqrtsize);
  std::vector<int> periodic(2, 0);
  std::vector<int> subDims(2);
  subDims = {0, 1};
  MPI_Comm cgrid;
  MPI_Comm ccolumn;
  MPI_Comm crow;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dimSize.data(), periodic.data(), 0, &cgrid);
  MPI_Cart_coords(cgrid, rank, 2, coordinates_of_grid.data());
  MPI_Cart_sub(cgrid, subDims.data(), &crow);
  std::swap(subDims[0], subDims[1]);
  MPI_Cart_sub(cgrid, subDims.data(), &ccolumn);

  int BSize = static_cast<int>(ceil(static_cast<double>(N) / sqrtsize));
  int BBSize = BSize * BSize;
  std::vector<double> pAblock(BBSize, 0);
  std::vector<double> pBblock(BBSize, 0);
  std::vector<double> pCblock(BBSize, 0);
  if (rank == 0) {
    for (int i = 0; i < BSize; i++)
      for (int j = 0; j < BSize; j++) {
        pAblock[i * BSize + j] = A[i * N + j];
        pBblock[i * BSize + j] = B[i * N + j];
      }
  }
  int enlarged_size = BSize * sqrtsize;
  MPI_Datatype typeofblock;
  MPI_Type_vector(BSize, BSize, enlarged_size, MPI_DOUBLE, &typeofblock);
  MPI_Type_commit(&typeofblock);

  MPI_Status Status;
  if (rank == 0) {
    for (int l = 1; l < size; l++) {
      MPI_Send(A.data() + (l % sqrtsize) * BSize + (l / sqrtsize) * N * BSize, 1, typeofblock, l, 0, cgrid);
      MPI_Send(B.data() + (l % sqrtsize) * BSize + (l / sqrtsize) * N * BSize, 1, typeofblock, l, 1, cgrid);
    }
  } else {
    MPI_Recv(pAblock.data(), BBSize, MPI_DOUBLE, 0, 0, cgrid, &Status);
    MPI_Recv(pBblock.data(), BBSize, MPI_DOUBLE, 0, 1, cgrid, &Status);
  }

  for (int i = 0; i < sqrtsize; i++) {
    std::vector<double> tmpmatra(BBSize);
    int bcast = (coordinates_of_grid[0] + i) % sqrtsize;
    if (coordinates_of_grid[1] == bcast) tmpmatra = pAblock;
    MPI_Bcast(tmpmatra.data(), BBSize, MPI_DOUBLE, bcast, crow);
    for (int j = 0; j < BSize; j++)
      for (int k = 0; k < BSize; k++) {
        double temp = 0;
        for (int l = 0; l < BSize; l++) temp += tmpmatra[j * BSize + l] * pBblock[l * BSize + k];
        pCblock[j * BSize + k] += temp;
      }
    int nextp = coordinates_of_grid[0] + 1;
    if (coordinates_of_grid[0] == sqrtsize - 1) nextp = 0;
    int prevp = coordinates_of_grid[0] - 1;
    if (coordinates_of_grid[0] == 0) prevp = sqrtsize - 1;
    MPI_Sendrecv_replace(pBblock.data(), BBSize, MPI_DOUBLE, prevp, 0, nextp, 0, ccolumn, &Status);
  }

  if (rank == 0) {
    for (int i = 0; i < BSize; i++)
      for (int j = 0; j < BSize; j++) C[i * N + j] = pCblock[i * BSize + j];
  }

  if (rank != 0) MPI_Send(pCblock.data(), BBSize, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
  if (rank == 0) {
    for (int i = 1; i < size; i++)
      MPI_Recv(C.data() + (i % sqrtsize) * BSize + (i / sqrtsize) * N * BSize, BBSize, typeofblock, i, 3,
               MPI_COMM_WORLD, &Status);
  }

  MPI_Type_free(&typeofblock);

  return true;
}

bool alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
    dense_matrix_multiplication_block_scheme_fox_algorithm_mpi::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* res = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(C.begin(), C.end(), res);
  }
  return true;
}