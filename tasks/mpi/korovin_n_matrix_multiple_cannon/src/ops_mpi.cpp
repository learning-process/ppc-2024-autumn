#include "mpi/korovin_n_matrix_multiple_cannon/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  numRowsA_ = taskData->inputs_count[0];
  numColsA_RowsB_ = taskData->inputs_count[1];
  numColsB_ = taskData->inputs_count[2];

  auto* a_data = reinterpret_cast<double*>(taskData->inputs[0]);
  A_.assign(a_data, a_data + (numRowsA_ * numColsA_RowsB_));
  auto* b_data = reinterpret_cast<double*>(taskData->inputs[1]);
  B_.assign(b_data, b_data + (numColsA_RowsB_ * numColsB_));

  return true;
}

bool korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.size() < 3) return false;

  int temp_numRowsA = taskData->inputs_count[0];
  int temp_numColsA_RowsB = taskData->inputs_count[1];
  int temp_numColsB = taskData->inputs_count[2];

  return (temp_numRowsA > 0 && temp_numColsA_RowsB > 0 && temp_numColsB > 0) &&
         (taskData->inputs.size() >= 2 && taskData->inputs[0] != nullptr && taskData->inputs[1] != nullptr) &&
         (!taskData->outputs.empty() && taskData->outputs[0] != nullptr);
}

bool korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  C_.resize(numRowsA_ * numColsB_);
  for (int i = 0; i < numRowsA_; i++) {
    for (int j = 0; j < numColsB_; j++) {
      for (int p = 0; p < numColsA_RowsB_; p++) {
        C_[i * numColsB_ + j] += A_[i * numColsA_RowsB_ + p] * B_[p * numColsB_ + j];
      }
    }
  }

  return true;
}

bool korovin_n_matrix_multiple_cannon_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  auto* data_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(C_.begin(), C_.end(), data_ptr);

  return true;
}

bool korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int rank;
  MPI_Comm_rank(world, &rank);
  if (rank == 0) {
    numRowsA_ = taskData->inputs_count[0];
    numColsA_RowsB_ = taskData->inputs_count[1];
    numColsB_ = taskData->inputs_count[2];
    auto* a_data = reinterpret_cast<double*>(taskData->inputs[0]);
    A_original_.assign(a_data, a_data + (numRowsA_ * numColsA_RowsB_));
    auto* b_data = reinterpret_cast<double*>(taskData->inputs[1]);
    B_original_.assign(b_data, b_data + (numColsA_RowsB_ * numColsB_));
  }

  return true;
}

bool korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  int rank;
  MPI_Comm_rank(world, &rank);
  if (rank != 0) return true;
  if (taskData->inputs_count.size() < 3) return false;

  int temp_numRowsA = taskData->inputs_count[0];
  int temp_numColsA_RowsB = taskData->inputs_count[1];
  int temp_numColsB = taskData->inputs_count[2];

  return (temp_numRowsA > 0 && temp_numColsA_RowsB > 0 && temp_numColsB > 0) &&
         (taskData->inputs.size() >= 2 && taskData->inputs[0] != nullptr && taskData->inputs[1] != nullptr) &&
         (!taskData->outputs.empty() && taskData->outputs[0] != nullptr);
}

bool korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int size;
  int rank;
  MPI_Comm_size(world, &size);
  MPI_Comm_rank(world, &rank);

  if (rank == 0) {
    A_ = A_original_;
    B_ = B_original_;
  }

  MPI_Bcast(&numRowsA_, 1, MPI_INT, 0, world);
  MPI_Bcast(&numColsA_RowsB_, 1, MPI_INT, 0, world);
  MPI_Bcast(&numColsB_, 1, MPI_INT, 0, world);

  int q = static_cast<int>(std::floor(std::sqrt(size)));
  int active_procs = q * q;

  int padded_m = q * ((numRowsA_ + q - 1) / q);
  int padded_n = q * ((numColsA_RowsB_ + q - 1) / q);
  int padded_k = q * ((numColsB_ + q - 1) / q);

  int block_m = padded_m / q;
  int block_n = padded_n / q;
  int block_k = padded_k / q;

  if (rank == 0) {
    std::vector<double> A_padded(padded_m * padded_n, 0.0);
    for (int i = 0; i < numRowsA_; i++) {
      std::copy(A_.begin() + i * numColsA_RowsB_, A_.begin() + (i + 1) * numColsA_RowsB_,
                A_padded.begin() + i * padded_n);
    }
    std::vector<double> B_padded(padded_n * padded_k, 0.0);
    for (int i = 0; i < numColsA_RowsB_; i++) {
      std::copy(B_.begin() + i * numColsB_, B_.begin() + (i + 1) * numColsB_, B_padded.begin() + i * padded_k);
    }
    A_ = std::move(A_padded);
    B_ = std::move(B_padded);
  }

  bool is_active = (rank < active_procs);
  int color = is_active ? 1 : MPI_UNDEFINED;
  MPI_Comm active_comm;
  MPI_Comm_split(world, color, rank, &active_comm);

  if (!is_active) {
    return true;
  }

  MPI_Comm cart_comm;
  int dims_grid[2] = {q, q};
  int periods_grid[2] = {1, 1};
  int reorder_grid = 0;
  MPI_Cart_create(active_comm, 2, dims_grid, periods_grid, reorder_grid, &cart_comm);

  int cart_rank;
  MPI_Comm_rank(cart_comm, &cart_rank);
  int coords[2];
  MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
  int row = coords[0];
  int col = coords[1];

  int left_rank;
  int right_rank;
  int up_rank;
  int down_rank;
  MPI_Cart_shift(cart_comm, 1, 1, &left_rank, &right_rank);
  MPI_Cart_shift(cart_comm, 0, 1, &up_rank, &down_rank);

  std::vector<double> A_local(block_m * block_n, 0.0);
  std::vector<double> B_local(block_n * block_k, 0.0);
  std::vector<double> C_local(block_m * block_k, 0.0);

  if (cart_rank == 0) {
    for (int p = 0; p < active_procs; ++p) {
      int p_coords[2];
      MPI_Cart_coords(cart_comm, p, 2, p_coords);
      int p_row = p_coords[0];
      int p_col = p_coords[1];

      std::vector<double> A_block(block_m * block_n, 0.0);
      for (int i = 0; i < block_m; i++) {
        int src_index = (p_row * block_m + i) * padded_n + (p_col * block_n);
        std::copy(A_.begin() + src_index, A_.begin() + src_index + block_n, A_block.begin() + i * block_n);
      }
      std::vector<double> B_block(block_n * block_k, 0.0);
      for (int i = 0; i < block_n; i++) {
        int src_index = (p_row * block_n + i) * padded_k + (p_col * block_k);
        std::copy(B_.begin() + src_index, B_.begin() + src_index + block_k, B_block.begin() + i * block_k);
      }
      if (p == 0) {
        A_local = std::move(A_block);
        B_local = std::move(B_block);
      } else {
        MPI_Send(A_block.data(), block_m * block_n, MPI_DOUBLE, p, 0, cart_comm);
        MPI_Send(B_block.data(), block_n * block_k, MPI_DOUBLE, p, 1, cart_comm);
      }
    }
  } else {
    MPI_Recv(A_local.data(), block_m * block_n, MPI_DOUBLE, 0, 0, cart_comm, MPI_STATUS_IGNORE);
    MPI_Recv(B_local.data(), block_n * block_k, MPI_DOUBLE, 0, 1, cart_comm, MPI_STATUS_IGNORE);
  }

  for (int i = 0; i < row; i++) {
    MPI_Sendrecv_replace(A_local.data(), block_m * block_n, MPI_DOUBLE, left_rank, 2, right_rank, 2, cart_comm,
                         MPI_STATUS_IGNORE);
  }
  for (int i = 0; i < col; i++) {
    MPI_Sendrecv_replace(B_local.data(), block_n * block_k, MPI_DOUBLE, up_rank, 3, down_rank, 3, cart_comm,
                         MPI_STATUS_IGNORE);
  }

  for (int iter = 0; iter < q; iter++) {
    for (int i = 0; i < block_m; i++) {
      for (int l = 0; l < block_n; l++) {
        double a_il = A_local[i * block_n + l];
        for (int j = 0; j < block_k; j++) {
          C_local[i * block_k + j] += a_il * B_local[l * block_k + j];
        }
      }
    }
    MPI_Sendrecv_replace(A_local.data(), block_m * block_n, MPI_DOUBLE, left_rank, 4, right_rank, 4, cart_comm,
                         MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(B_local.data(), block_n * block_k, MPI_DOUBLE, up_rank, 5, down_rank, 5, cart_comm,
                         MPI_STATUS_IGNORE);
  }

  if (cart_rank != 0) {
    MPI_Send(C_local.data(), block_m * block_k, MPI_DOUBLE, 0, 6, cart_comm);
  } else {
    C_.resize(padded_m * padded_k, 0.0);
    for (int p = 0; p < active_procs; p++) {
      if (p == 0) {
        for (int i = 0; i < block_m; i++) {
          int dest_index = i * padded_k;
          std::copy(C_local.begin() + i * block_k, C_local.begin() + (i + 1) * block_k, C_.begin() + dest_index);
        }
      } else {
        std::vector<double> recv_C(block_m * block_k);
        MPI_Recv(recv_C.data(), block_m * block_k, MPI_DOUBLE, p, 6, cart_comm, MPI_STATUS_IGNORE);

        int p_coords[2];
        MPI_Cart_coords(cart_comm, p, 2, p_coords);
        int p_row = p_coords[0];
        int p_col = p_coords[1];

        int start_row = p_row * block_m;
        int start_col = p_col * block_k;

        for (int i = 0; i < block_m; i++) {
          int dest_row = start_row + i;
          int dest_index = dest_row * padded_k + start_col;
          std::copy(recv_C.begin() + i * block_k, recv_C.begin() + (i + 1) * block_k, C_.begin() + dest_index);
        }
      }
    }

    std::vector<double> C_final(numRowsA_ * numColsB_, 0.0);
    for (int i = 0; i < numRowsA_; i++) {
      std::copy(C_.begin() + i * padded_k, C_.begin() + i * padded_k + numColsB_, C_final.begin() + i * numColsB_);
    }
    C_ = std::move(C_final);
  }

  if (cart_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&cart_comm);
  }
  MPI_Comm_free(&active_comm);

  return true;
}

bool korovin_n_matrix_multiple_cannon_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  int rank;
  MPI_Comm_rank(world, &rank);
  if (rank == 0) {
    auto* data_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(C_.begin(), C_.end(), data_ptr);
  }

  return true;
}
