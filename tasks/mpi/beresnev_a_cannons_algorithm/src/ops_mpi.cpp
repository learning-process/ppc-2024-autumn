// Copyright 2023 Nesterov Alexander
#include "mpi/beresnev_a_cannons_algorithm/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

static int find_compatible_q(int size, int N) {
  int q = std::floor(std::sqrt(size));
  while (q > 0) {
    if (N % q == 0) {
      break;
    }
    --q;
  }
  return q > 0 ? q : 1;
}

static void extract_block(const std::vector<double>& matrix, double* block, int N, int K, int block_row,
                          int block_col) {
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      block[i * K + j] = matrix[(block_row * K + i) * N + (block_col * K + j)];
    }
  }
}

static void multiply_blocks(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int K) {
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i * K + j] += A[i * K + k] * B[k * K + j];
      }
    }
  }
}

static void rearrange_matrix(const std::vector<double>& gathered_blocks, std::vector<double>& final_matrix, int N,
                             int K, int q) {
  for (int block_row = 0; block_row < q; ++block_row) {
    for (int block_col = 0; block_col < q; ++block_col) {
      int block_rank = block_row * q + block_col;
      int block_index = block_rank * K * K;

      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
          int global_row = block_row * K + i;
          int global_col = block_col * K + j;
          final_matrix[global_row * N + global_col] = gathered_blocks[block_index + i * K + j];
        }
      }
    }
  }
}

bool beresnev_a_cannons_algorithm_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  auto* A = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* B = reinterpret_cast<double*>(taskData->inputs[1]);
  inp_A.assign(A, A + s_);
  inp_B.assign(B, B + s_);
  res_.resize(s_);
  return true;
}

bool beresnev_a_cannons_algorithm_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs[2] != nullptr) {
    n_ = reinterpret_cast<int*>(taskData->inputs[2])[0];
  }
  s_ = n_ * n_;
  return n_ > 0 && taskData->inputs_count[2] == 1 && taskData->inputs_count[0] == taskData->inputs_count[1] &&
         static_cast<int>(taskData->inputs_count[1]) == s_ && taskData->inputs[0] != nullptr &&
         taskData->inputs[1] != nullptr && taskData->outputs[0] != nullptr &&
         static_cast<int>(taskData->outputs_count[0]) == s_;
}

bool beresnev_a_cannons_algorithm_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      for (int k = 0; k < n_; ++k) {
        res_[i * n_ + j] += inp_A[i * n_ + k] * inp_B[k * n_ + j];
      }
    }
  }
  return true;
}

bool beresnev_a_cannons_algorithm_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0].assign(res_.data(), res_.data() + s_);
  return true;
}

bool beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* A = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* B = reinterpret_cast<double*>(taskData->inputs[1]);
    inp_A.assign(A, A + s_);
    inp_B.assign(B, B + s_);
    res_ = std::vector<double>(s_);
  }
  return true;
}

bool beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs[2] != nullptr) {
      n_ = reinterpret_cast<int*>(taskData->inputs[2])[0];
    }
    s_ = n_ * n_;
    return n_ > 0 && taskData->inputs_count[2] == 1 && taskData->inputs_count[0] == taskData->inputs_count[1] &&
           static_cast<int>(taskData->inputs_count[1]) == s_ && taskData->inputs[0] != nullptr &&
           taskData->inputs[1] != nullptr && taskData->outputs[0] != nullptr &&
           static_cast<int>(taskData->outputs_count[0]) == s_;
  }
  return true;
}

bool beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int rank = world.rank();
  int size = world.size();

  boost::mpi::broadcast(world, n_, 0);
  boost::mpi::broadcast(world, s_, 0);

  int q = find_compatible_q(size, n_);
  int K = n_ / q;

  int color = (rank < q * q) ? 1 : MPI_UNDEFINED;

  MPI_Comm new_comm;
  MPI_Comm_split(world, color, rank, &new_comm);

  if (color == MPI_UNDEFINED) {
    return true;
  }

  boost::mpi::communicator my_world(new_comm, boost::mpi::comm_take_ownership);
  rank = my_world.rank();
  size = my_world.size();

  std::vector<double> scatter_A(s_);
  std::vector<double> scatter_B(s_);
  if (rank == 0) {
    int index = 0;
    for (int block_row = 0; block_row < q; ++block_row) {
      for (int block_col = 0; block_col < q; ++block_col) {
        extract_block(inp_A, scatter_A.data() + index, n_, K, block_row, block_col);
        extract_block(inp_B, scatter_B.data() + index, n_, K, block_row, block_col);
        index += K * K;
      }
    }
  }

  std::vector<double> local_A(K * K);
  std::vector<double> local_B(K * K);
  std::vector<double> local_C(K * K, 0.0);
  std::vector<double> unfinished_C(s_);

  boost::mpi::scatter(my_world, scatter_A, local_A.data(), K * K, 0);
  boost::mpi::scatter(my_world, scatter_B, local_B.data(), K * K, 0);

  int row = rank / q;
  int col = rank % q;

  int send_rank_A = row * q + (col + q - 1) % q;
  int recv_rank_A = row * q + (col + 1) % q;

  if (send_rank_A >= size || recv_rank_A >= size) {
    std::cerr << "Invalid rank for send or receive: send_rank=" << send_rank_A << ", recv_rank=" << recv_rank_A
              << std::endl;
    return false;
  }

  int send_rank_B = col + q * ((row + q - 1) % q);
  int recv_rank_B = col + q * ((row + 1) % q);

  if (send_rank_B >= size || recv_rank_B >= size) {
    std::cerr << "Invalid rank for send or receive: send_rank=" << send_rank_B << ", recv_rank=" << recv_rank_B
              << std::endl;
    return false;
  }

  for (int i = 0; i < row; ++i) {
    boost::mpi::request send_request;
    boost::mpi::request recv_request;

    std::vector<double> temp(local_A.size());
    send_request = my_world.isend(send_rank_A, 0, local_A.data(), local_A.size());
    recv_request = my_world.irecv(recv_rank_A, 0, temp.data(), temp.size());

    if (send_request.active() && recv_request.active()) {
      send_request.wait();
      recv_request.wait();
    } else {
      return false;
    }

    local_A = temp;
  }

  for (int i = 0; i < col; ++i) {
    boost::mpi::request send_request1;
    boost::mpi::request recv_request1;

    std::vector<double> temp_B(local_B.size());
    send_request1 = my_world.isend(send_rank_B, 1, local_B.data(), local_B.size());
    recv_request1 = my_world.irecv(recv_rank_B, 1, temp_B.data(), temp_B.size());

    if (send_request1.active() && recv_request1.active()) {
      send_request1.wait();
      recv_request1.wait();
    } else {
      return false;
    }

    local_B = temp_B;
  }

  multiply_blocks(local_A, local_B, local_C, K);

  for (int iter = 0; iter < q - 1; ++iter) {
    boost::mpi::request send_request2;
    boost::mpi::request recv_request2;

    boost::mpi::request send_request3;
    boost::mpi::request recv_request3;

    std::vector<double> temp_A(local_A.size());
    send_request2 = my_world.isend(send_rank_A, 0, local_A.data(), local_A.size());
    recv_request2 = my_world.irecv(recv_rank_A, 0, temp_A.data(), temp_A.size());

    std::vector<double> temp_B(local_B.size());
    send_request3 = my_world.isend(send_rank_B, 1, local_B.data(), local_B.size());
    recv_request3 = my_world.irecv(recv_rank_B, 1, temp_B.data(), temp_B.size());

    if (send_request2.active() && recv_request2.active() && send_request3.active() && recv_request3.active()) {
      send_request2.wait();
      recv_request2.wait();
      send_request3.wait();
      recv_request3.wait();
    } else {
      return false;
    }

    local_A = temp_A;
    local_B = temp_B;

    multiply_blocks(local_A, local_B, local_C, K);
  }

  boost::mpi::gather(my_world, local_C.data(), local_C.size(), unfinished_C, 0);
  if (rank == 0) {
    rearrange_matrix(unfinished_C, res_, n_, K, q);
  }
  return true;
}

bool beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<std::vector<double>*>(taskData->outputs[0])[0].assign(res_.data(), res_.data() + s_);
  }
  return true;
}