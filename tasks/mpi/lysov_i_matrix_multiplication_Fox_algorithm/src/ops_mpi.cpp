// Copyright 2023 Nesterov Alexander
#include "mpi/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <random>
#include <vector>

static void extract_submatrix_block(const std::vector<double>& matrix, double* block, int total_columns, int block_size,
                                    int block_row_index, int block_col_index) {
  for (int row = 0; row < block_size; ++row) {
    for (int col = 0; col < block_size; ++col) {
      block[row * block_size + col] =
          matrix[(block_row_index * block_size + row) * total_columns + (block_col_index * block_size + col)];
    }
  }
}

static void multiply_matrix_blocks(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C,
                                   int block_size) {
  for (int row = 0; row < block_size; ++row) {
    for (int col = 0; col < block_size; ++col) {
      double sum = 0.0;
      for (int k = 0; k < block_size; ++k) {
        sum += A[row * block_size + k] * B[k * block_size + col];
      }
      C[row * block_size + col] += sum;
    }
  }
}

static void perform_fox_algorithm_step(boost::mpi::communicator& my_world, int rank, int cnt_work_process, int K,
                                       std::vector<double>& local_A, std::vector<double>& local_B,
                                       std::vector<double>& local_C) {
  std::vector<double> temp_A(K * K);
  std::vector<double> temp_B(K * K);

  for (int l = 0; l < cnt_work_process; ++l) {
    boost::mpi::request send_request1;
    boost::mpi::request recv_request1;
    boost::mpi::request send_request2;
    boost::mpi::request recv_request2;

    int row = rank / cnt_work_process;
    int col = rank % cnt_work_process;

    if (col == (row + l) % cnt_work_process) {
      for (int target_col = 0; target_col < cnt_work_process; ++target_col) {
        if (target_col != col) {
          int target_rank = row * cnt_work_process + target_col;
          auto request = my_world.isend(target_rank, 0, local_A.data(), K * K);
          request.wait();
        }
      }
      temp_A = local_A;
    } else {
      int sender_rank = row * cnt_work_process + ((row + l) % cnt_work_process);
      auto request = my_world.irecv(sender_rank, 0, temp_A.data(), K * K);
      request.wait();
    }
    send_request1.wait();
    recv_request1.wait();
    my_world.barrier();

    multiply_matrix_blocks(temp_A, local_B, local_C, K);

    int send_to = ((row - 1 + cnt_work_process) % cnt_work_process) * cnt_work_process + col;
    int recv_from = ((row + 1) % cnt_work_process) * cnt_work_process + col;

    send_request2 = my_world.isend(send_to, 0, local_B.data(), K * K);
    recv_request2 = my_world.irecv(recv_from, 0, temp_B.data(), K * K);
    my_world.barrier();
    send_request2.wait();
    recv_request2.wait();

    local_B = temp_B;
  }
}

bool lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  N = reinterpret_cast<std::size_t*>(taskData->inputs[0])[0];
  block_size = reinterpret_cast<std::size_t*>(taskData->inputs[3])[0];
  A.resize(N * N);
  B.resize(N * N);
  C.resize(N * N, 0.0);
  std::copy(reinterpret_cast<double*>(taskData->inputs[1]), reinterpret_cast<double*>(taskData->inputs[1]) + N * N,
            A.begin());
  std::copy(reinterpret_cast<double*>(taskData->inputs[2]), reinterpret_cast<double*>(taskData->inputs[2]) + N * N,
            B.begin());
  return true;
}

bool lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  N = reinterpret_cast<int*>(taskData->inputs[0])[0];
  block_size = reinterpret_cast<std::size_t*>(taskData->inputs[3])[0];
  if (taskData->inputs_count.size() != 4 || taskData->outputs_count.size() != 1) {
    return false;
  }
  if (taskData->inputs_count[1] != static_cast<uint32_t>(N * N) ||
      taskData->inputs_count[2] != static_cast<uint32_t>(N * N)) {
    return false;
  }
  return taskData->outputs_count[0] == static_cast<uint32_t>(N * N) && block_size > 0;
}

bool lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < N; ++k) {
        double a_ij = A[i * N + k];
        double b_kj = B[k * N + j];
        sum += a_ij * b_kj;
      }
      C[i * N + j] = sum;
    }
  }
  return true;
}

bool lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::copy(C.begin(), C.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}

bool lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* A = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* B = reinterpret_cast<double*>(taskData->inputs[1]);
    initialMatrixA.assign(A, A + elements);
    initialMatrixB.assign(B, B + elements);
    resultC = std::vector<double>(elements);
  }
  return true;
}

bool lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    dimension = *reinterpret_cast<int*>(taskData->inputs[2]);
    elements = dimension * dimension;
    if (dimension <= 0) {
      return false;
    }
    if (taskData->inputs_count[2] != sizeof(int)) {
      return false;
    }

    if (taskData->inputs_count[0] != taskData->inputs_count[1]) {
      return false;
    }

    if (static_cast<int>(taskData->inputs_count[1]) != elements) {
      return false;
    }
    if (taskData->inputs[0] == nullptr || taskData->inputs[1] == nullptr || taskData->outputs[0] == nullptr) {
      return false;
    }
    if (static_cast<int>(taskData->outputs_count[0]) != elements) {
      return false;
    }
  }
  return true;
}

bool lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int rank = world.rank();
  int size = world.size();

  boost::mpi::broadcast(world, dimension, 0);
  boost::mpi::broadcast(world, elements, 0);

  int cnt_work_process = std::floor(std::sqrt(size));
  while (cnt_work_process > 0) {
    if (size % cnt_work_process == 0) {
      break;
    }
    --cnt_work_process;
  }
  if (cnt_work_process <= 0) {
    cnt_work_process = 1;
  }
  int K = dimension / cnt_work_process;
  int process_group = (rank < cnt_work_process * cnt_work_process) ? 1 : MPI_UNDEFINED;
  MPI_Comm computation_comm;
  MPI_Comm_split(world, process_group, rank, &computation_comm);

  if (process_group == MPI_UNDEFINED) {
    return true;
  }

  boost::mpi::communicator my_communicator(computation_comm, boost::mpi::comm_take_ownership);
  rank = my_communicator.rank();

  std::vector<double> scatter_A(elements);
  std::vector<double> scatter_B(elements);
  if (rank == 0) {
    int index = 0;
    for (int block_row = 0; block_row < cnt_work_process; ++block_row) {
      for (int block_col = 0; block_col < cnt_work_process; ++block_col) {
        extract_submatrix_block(initialMatrixA, scatter_A.data() + index, dimension, K, block_row, block_col);
        extract_submatrix_block(initialMatrixB, scatter_B.data() + index, dimension, K, block_row, block_col);
        index += K * K;
      }
    }
  }
  std::vector<double> local_matrixA(K * K);
  boost::mpi::scatter(my_communicator, scatter_A, local_matrixA.data(), K * K, 0);
  std::vector<double> local_matrixB(K * K);
  boost::mpi::scatter(my_communicator, scatter_B, local_matrixB.data(), K * K, 0);
  std::vector<double> local_matrixC(K * K, 0.0);
  std::vector<double> unfinished_C(elements);

  perform_fox_algorithm_step(my_communicator, rank, cnt_work_process, K, local_matrixA, local_matrixB, local_matrixC);
  boost::mpi::gather(my_communicator, local_matrixC.data(), local_matrixC.size(), unfinished_C, 0);

  if (rank == 0) {
    for (int block_row = 0; block_row < cnt_work_process; ++block_row) {
      for (int block_col = 0; block_col < cnt_work_process; ++block_col) {
        int block_rank = block_row * cnt_work_process + block_col;
        int block_index = block_rank * K * K;

        for (int i = 0; i < K; ++i) {
          for (int j = 0; j < K; ++j) {
            int global_row = block_row * K + i;
            int global_col = block_col * K + j;
            resultC[global_row * dimension + global_col] = unfinished_C[block_index + i * K + j];
          }
        }
      }
    }
  }
  return true;
}

bool lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0].assign(resultC.data(), resultC.data() + elements);
  }
  return true;
}