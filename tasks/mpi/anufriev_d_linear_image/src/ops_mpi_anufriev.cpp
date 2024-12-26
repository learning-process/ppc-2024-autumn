#include "mpi/anufriev_d_linear_image/include/ops_mpi_anufriev.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <iostream>

namespace anufriev_d_linear_image {

SimpleIntMPI::SimpleIntMPI(const std::shared_ptr<ppc::core::TaskData>& taskData) : Task(taskData) {}

bool SimpleIntMPI::pre_processing() {
  internal_order_test();
  return true;
}

bool SimpleIntMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (!taskData || taskData->inputs.size() < 3 || taskData->inputs_count.size() < 3 || taskData->outputs.empty() ||
        taskData->outputs_count.empty()) {
      std::cerr << "Validation failed: Недостаточно входных или выходных данных.\n";
      return false;
    }

    width_ = *reinterpret_cast<int*>(taskData->inputs[1]);
    height_ = *reinterpret_cast<int*>(taskData->inputs[2]);

    auto expected_size = static_cast<size_t>(width_ * height_ * sizeof(int));

    if (width_ < 3 || height_ < 3) {
      std::cerr << "Validation failed: width или height меньше 3.\n";
      return false;
    }

    if (taskData->inputs_count[0] != expected_size) {
      std::cerr << "Validation failed: inputs_count[0] != width * height * sizeof(int).\n";
      std::cerr << "Expected: " << expected_size << ", Got: " << taskData->inputs_count[0] << "\n";
      return false;
    }

    if (taskData->outputs_count[0] != expected_size) {
      std::cerr << "Validation failed: outputs_count[0] != width * height * sizeof(int).\n";
      std::cerr << "Expected: " << expected_size << ", Got: " << taskData->outputs_count[0] << "\n";
      return false;
    }

    original_data_.resize(width_ * height_);
    int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(input_ptr, input_ptr + (width_ * height_), original_data_.begin());
  }

  boost::mpi::broadcast(world, width_, 0);
  boost::mpi::broadcast(world, height_, 0);
  total_size_ = static_cast<size_t>(width_ * height_);

  return true;
}

bool SimpleIntMPI::run() {
  internal_order_test();

  distributeData();
  exchangeHalo();
  applyGaussianFilter();
  gatherData();

  return true;
}

bool SimpleIntMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(processed_data_.begin(), processed_data_.end(), output_ptr);
  }
  return true;
}

void SimpleIntMPI::distributeData() {
  MPI_Comm comm = world;
  int nprocs = world.size();
  int rank = world.rank();

  int base_cols = width_ / nprocs;
  int remainder = width_ % nprocs;

  std::vector<int> sendcounts(nprocs);
  std::vector<int> displs(nprocs);

  for (int i = 0; i < nprocs; ++i) {
    sendcounts[i] = (base_cols + (i < remainder ? 1 : 0)) * height_;
    displs[i] = (i < remainder) ? i * (base_cols + 1) * height_
                                : remainder * (base_cols + 1) * height_ + (i - remainder) * base_cols * height_;
  }

  local_width_ = base_cols + (rank < remainder ? 1 : 0);
  start_col_ =
      (rank < remainder) ? rank * (base_cols + 1) : remainder * (base_cols + 1) + (rank - remainder) * base_cols;

  int halo_cols = 2;
  local_data_.resize((local_width_ + halo_cols) * height_, 0);

  std::vector<int> transposed_original;
  if (rank == 0) {
    transposed_original.resize(width_ * height_);
    for (int r = 0; r < height_; ++r) {
      for (int c = 0; c < width_; ++c) {
        transposed_original[c * height_ + r] = original_data_[r * width_ + c];
      }
    }
  }

  MPI_Scatterv(world.rank() == 0 ? transposed_original.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
               &local_data_[height_], local_width_ * height_, MPI_INT, 0, comm);
}

void SimpleIntMPI::exchangeHalo() {
  MPI_Comm comm = world;
  int rank = world.rank();
  int nprocs = world.size();

  int left = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
  int right = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

  std::vector<int> send_left(height_);
  std::vector<int> send_right(height_);
  std::vector<int> recv_left(height_);
  std::vector<int> recv_right(height_);

  if (local_width_ > 0) {
    std::copy(&local_data_[height_], &local_data_[2 * height_], send_left.begin());
    std::copy(&local_data_[(local_width_)*height_], &local_data_[(local_width_ + 1) * height_], send_right.begin());
  }

  if (left != MPI_PROC_NULL) {
    MPI_Sendrecv(send_left.data(), height_, MPI_INT, left, 0, recv_left.data(), height_, MPI_INT, left, 1, comm,
                 MPI_STATUS_IGNORE);
  } else {
    std::copy(send_left.begin(), send_left.end(), recv_left.begin());
  }

  if (right != MPI_PROC_NULL) {
    MPI_Sendrecv(send_right.data(), height_, MPI_INT, right, 1, recv_right.data(), height_, MPI_INT, right, 0, comm,
                 MPI_STATUS_IGNORE);
  } else {
    std::copy(send_right.begin(), send_right.end(), recv_right.begin());
  }

  if (left != MPI_PROC_NULL) {
    std::copy(recv_left.begin(), recv_left.end(), local_data_.begin());
  } else {
    std::copy(&local_data_[height_], &local_data_[2 * height_], local_data_.begin());
  }

  if (right != MPI_PROC_NULL) {
    std::copy(recv_right.begin(), recv_right.end(), &local_data_[(local_width_ + 1) * height_]);
  } else {
    std::copy(&local_data_[(local_width_)*height_], &local_data_[(local_width_ + 1) * height_],
              &local_data_[(local_width_ + 1) * height_]);
  }
}

void SimpleIntMPI::applyGaussianFilter() {
  std::vector<int> result(local_width_ * height_, 0);

  for (int c = 1; c <= local_width_; c++) {
    for (int r = 0; r < height_; r++) {
      int sum = 0;
      for (int kc = -1; kc <= 1; kc++) {
        for (int kr = -1; kr <= 1; kr++) {
          int cc = c + kc;
          int rr = std::min(std::max(r + kr, 0), height_ - 1);

          sum += local_data_[cc * height_ + rr] * kernel_[kr + 1][kc + 1];
        }
      }
      result[(c - 1) * height_ + r] = sum / 16;
    }
  }

  std::copy(result.begin(), result.end(), &local_data_[height_]);
}

void SimpleIntMPI::gatherData() {
  MPI_Comm comm = world;
  int nprocs = world.size();

  int base_cols = width_ / nprocs;
  int remainder = width_ % nprocs;

  std::vector<int> recvcounts(nprocs);
  std::vector<int> displs(nprocs);

  for (int i = 0; i < nprocs; ++i) {
    recvcounts[i] = (base_cols + (i < remainder ? 1 : 0)) * height_;
    displs[i] = (i < remainder) ? i * (base_cols + 1) * height_
                                : remainder * (base_cols + 1) * height_ + (i - remainder) * base_cols * height_;
  }

  std::vector<int> gathered_transposed;
  if (world.rank() == 0) {
    gathered_transposed.resize(width_ * height_);
  }

  MPI_Gatherv(&local_data_[height_], local_width_ * height_, MPI_INT,
              world.rank() == 0 ? gathered_transposed.data() : nullptr, recvcounts.data(), displs.data(), MPI_INT, 0,
              comm);

  if (world.rank() == 0) {
    processed_data_.resize(width_ * height_);
    for (int r = 0; r < height_; ++r) {
      for (int c = 0; c < width_; ++c) {
        processed_data_[r * width_ + c] = gathered_transposed[c * height_ + r];
      }
    }
  }
}

const std::vector<int>& SimpleIntMPI::getDataPath() const { return data_path_; }

}  // namespace anufriev_d_linear_image