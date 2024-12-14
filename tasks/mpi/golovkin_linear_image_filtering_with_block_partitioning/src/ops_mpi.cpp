// Golovkin Maksim Task#3
#include "mpi/golovkin_linear_image_filtering_with_block_partitioning/include/ops_mpi.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace golovkin_linear_image_filtering_with_block_partitioning;

SimpleBlockMPI::SimpleBlockMPI(const std::shared_ptr<ppc::core::TaskData>& taskData) : ppc::core::Task(taskData) {}

bool SimpleBlockMPI::pre_processing() { return true; }

bool SimpleBlockMPI::validation() {
  if (world.rank() == 0) {
    if (!taskData || taskData->inputs.size() < 3 || taskData->outputs.empty()) {
      return false;
    }

    width_ = *reinterpret_cast<int*>(taskData->inputs[1]);
    height_ = *reinterpret_cast<int*>(taskData->inputs[2]);

    if (width_ < 3 || height_ < 3) {
      return false;
    }

    original_data_.resize(width_ * height_);
    int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(input_ptr, input_ptr + (width_ * height_), original_data_.begin());
  }

  boost::mpi::broadcast(world, width_, 0);
  boost::mpi::broadcast(world, height_, 0);
  total_size_ = width_ * height_;

  return true;
}

bool SimpleBlockMPI::run() {
  distributeData();
  exchangeHalo();
  applyGaussianFilter();
  gatherData();
  return true;
}

bool SimpleBlockMPI::post_processing() {
  if (world.rank() == 0) {
    int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(processed_data_.begin(), processed_data_.end(), output_ptr);
  }
  return true;
}

void SimpleBlockMPI::distributeData() {
  int nprocs = world.size();
  int rank = world.rank();

  int base_rows = height_ / nprocs;
  int remainder = height_ % nprocs;

  local_height_ = base_rows + (rank < remainder ? 1 : 0);
  start_row_ = rank * base_rows + std::min(rank, remainder);

  local_data_.resize(local_height_ * width_);
  std::vector<int> sendcounts(nprocs, base_rows * width_);
  std::vector<int> displs(nprocs);

  for (int i = 0; i < remainder; ++i) {
    sendcounts[i] += width_;
  }

  for (int i = 0; i < nprocs; ++i) {
    displs[i] = i * base_rows * width_ + std::min(i, remainder) * width_;
  }

  MPI_Scatterv(original_data_.data(), sendcounts.data(), displs.data(), MPI_INT, local_data_.data(),
               local_height_ * width_, MPI_INT, 0, world);

  extended_local_height_ = local_height_;
}

void SimpleBlockMPI::exchangeHalo() {
  int nprocs = world.size();
  int rank = world.rank();

  assert(rank >= 0 && rank < nprocs);

  int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
  int down = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

  std::vector<int> send_up(width_, 0);
  std::vector<int> send_down(width_, 0);
  std::vector<int> recv_up(width_, 0);
  std::vector<int> recv_down(width_, 0);

  if (local_height_ > 0) {
    std::copy(local_data_.begin(), local_data_.begin() + width_, send_up.begin());
    std::copy(local_data_.end() - width_, local_data_.end(), send_down.begin());
  }

  MPI_Request send_up_req = MPI_REQUEST_NULL, recv_up_req = MPI_REQUEST_NULL;
  MPI_Request send_down_req = MPI_REQUEST_NULL, recv_down_req = MPI_REQUEST_NULL;

  if (up != MPI_PROC_NULL) {
    MPI_Isend(send_up.data(), width_, MPI_INT, up, 0, world, &send_up_req);
    MPI_Irecv(recv_up.data(), width_, MPI_INT, up, 1, world, &recv_up_req);
  }

  if (down != MPI_PROC_NULL) {
    MPI_Isend(send_down.data(), width_, MPI_INT, down, 1, world, &send_down_req);
    MPI_Irecv(recv_down.data(), width_, MPI_INT, down, 0, world, &recv_down_req);
  }

  MPI_Request requests[4] = {send_up_req, recv_up_req, send_down_req, recv_down_req};
  MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

  if (up != MPI_PROC_NULL && local_height_ > 0) {
    local_data_.insert(local_data_.begin(), recv_up.begin(), recv_up.end());
    extended_local_height_++;
  }

  if (down != MPI_PROC_NULL && local_height_ > 0) {
    local_data_.insert(local_data_.end(), recv_down.begin(), recv_down.end());
    extended_local_height_++;
  }
}

void SimpleBlockMPI::applyGaussianFilter() {
  std::vector<int> result(local_height_ * width_, 0);

  int halo_up = (world.rank() > 0) ? 1 : 0;

  const int kernel_sum = 16;

  for (int r = 0; r < local_height_; r++) {
    for (int c = 0; c < width_; c++) {
      int sum = 0;

      for (int kr = -1; kr <= 1; kr++) {
        for (int kc = -1; kc <= 1; kc++) {
          int rr = r + kr + halo_up;
          int cc = c + kc;

          if (rr >= 0 && rr < extended_local_height_ && cc >= 0 && cc < width_) {
            sum += local_data_[rr * width_ + cc] * kernel_[kr + 1][kc + 1];
          }
        }
      }

      result[r * width_ + c] = sum / kernel_sum;
    }
  }

  for (int r = 0; r < local_height_; r++) {
    std::copy(result.begin() + r * width_, result.begin() + (r + 1) * width_,
              local_data_.begin() + (r + halo_up) * width_);
  }
}

void SimpleBlockMPI::gatherData() {
  int nprocs = world.size();
  int rank = world.rank();

  int base_rows = height_ / nprocs;
  int remainder = height_ % nprocs;

  std::vector<int> recvcounts(nprocs, base_rows * width_);
  std::vector<int> displs(nprocs);

  for (int i = 0; i < remainder; ++i) {
    recvcounts[i] += width_;
  }

  for (int i = 0; i < nprocs; ++i) {
    displs[i] = i * base_rows * width_ + std::min(i, remainder) * width_;
  }

  if (rank == 0) {
    processed_data_.resize(width_ * height_);
  }

  std::vector<int> send_buffer(local_height_ * width_);
  int halo_up = (rank > 0) ? 1 : 0;
  for (int r = 0; r < local_height_; r++) {
    std::copy(local_data_.begin() + (r + halo_up) * width_, local_data_.begin() + (r + halo_up + 1) * width_,
              send_buffer.begin() + r * width_);
  }

  MPI_Gatherv(send_buffer.data(), local_height_ * width_, MPI_INT, rank == 0 ? processed_data_.data() : nullptr,
              recvcounts.data(), displs.data(), MPI_INT, 0, world);
}

const std::vector<int>& SimpleBlockMPI::getDataPath() const { return data_path_; }