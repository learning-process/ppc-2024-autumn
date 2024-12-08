#include "mpi/anufriev_d_linear_image/include/ops_mpi_anufriev.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <numeric>

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

    size_t expected_size = static_cast<size_t>(width_ * height_ * sizeof(int));

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

  int base_rows = height_ / nprocs;
  int remainder = height_ % nprocs;

  std::vector<int> sendcounts(nprocs);
  std::vector<int> displs(nprocs);

  for (int i = 0; i < nprocs; ++i) {
    sendcounts[i] = (base_rows + (i < remainder ? 1 : 0)) * width_;
    displs[i] = (i < remainder) ? i * (base_rows + 1) * width_
                                : remainder * (base_rows + 1) * width_ + (i - remainder) * base_rows * width_;
  }

  local_height_ = base_rows + (rank < remainder ? 1 : 0);
  start_row_ =
      (rank < remainder) ? rank * (base_rows + 1) : remainder * (base_rows + 1) + (rank - remainder) * base_rows;

  int halo_rows = 2;
  local_data_.resize((local_height_ + halo_rows) * width_, 0);

  MPI_Scatterv(world.rank() == 0 ? original_data_.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
               &local_data_[width_], local_height_ * width_, MPI_INT, 0, comm);
}

void SimpleIntMPI::exchangeHalo() {
  MPI_Comm comm = world;
  int rank = world.rank();
  int nprocs = world.size();

  int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
  int down = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

  std::vector<int> send_up(width_);
  std::vector<int> send_down(width_);
  std::vector<int> recv_up(width_);
  std::vector<int> recv_down(width_);

  if (local_height_ > 0) {
    std::copy(&local_data_[width_], &local_data_[2 * width_], send_up.begin());
    std::copy(&local_data_[(local_height_)*width_], &local_data_[(local_height_ + 1) * width_], send_down.begin());
  }

  MPI_Request reqs[4];
  int req_count = 0;

  if (up != MPI_PROC_NULL) {
    MPI_Isend(send_up.data(), width_, MPI_INT, up, 0, comm, &reqs[req_count++]);
    MPI_Irecv(recv_up.data(), width_, MPI_INT, up, 1, comm, &reqs[req_count++]);
  } else {
    std::copy(send_up.begin(), send_up.end(), recv_up.begin());
  }

  if (down != MPI_PROC_NULL) {
    MPI_Isend(send_down.data(), width_, MPI_INT, down, 1, comm, &reqs[req_count++]);
    MPI_Irecv(recv_down.data(), width_, MPI_INT, down, 0, comm, &reqs[req_count++]);
  } else {
    std::copy(send_down.begin(), send_down.end(), recv_down.begin());
  }

  if (req_count > 0) {
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
  }

  if (up != MPI_PROC_NULL) {
    std::copy(recv_up.begin(), recv_up.end(), local_data_.begin());
  } else {
    std::copy(send_up.begin(), send_up.end(), local_data_.begin());
  }

  if (down != MPI_PROC_NULL) {
    std::copy(recv_down.begin(), recv_down.end(), &local_data_[(local_height_ + 1) * width_]);
  } else {
    std::copy(send_down.begin(), send_down.end(), &local_data_[(local_height_ + 1) * width_]);
  }
}

void SimpleIntMPI::applyGaussianFilter() {
  std::vector<int> result(local_height_ * width_, 0);

  for (int r = 1; r <= local_height_; r++) {
    for (int c = 0; c < width_; c++) {
      int sum = 0;
      for (int kr = -1; kr <= 1; kr++) {
        for (int kc = -1; kc <= 1; kc++) {
          int rr = r + kr;
          int cc = std::min(std::max(c + kc, 0), width_ - 1);

          sum += local_data_[rr * width_ + cc] * kernel_[kr + 1][kc + 1];
        }
      }
      result[(r - 1) * width_ + c] = sum / 16;
    }
  }

  std::copy(result.begin(), result.end(), &local_data_[width_]);
}

void SimpleIntMPI::gatherData() {
  MPI_Comm comm = world;
  int nprocs = world.size();

  int base_rows = height_ / nprocs;
  int remainder = height_ % nprocs;

  std::vector<int> recvcounts(nprocs);
  std::vector<int> displs(nprocs);

  for (int i = 0; i < nprocs; ++i) {
    recvcounts[i] = (base_rows + (i < remainder ? 1 : 0)) * width_;
    displs[i] = (i < remainder) ? i * (base_rows + 1) * width_
                                : remainder * (base_rows + 1) * width_ + (i - remainder) * base_rows * width_;
  }

  if (world.rank() == 0) {
    processed_data_.resize(width_ * height_);
  }

  MPI_Gatherv(&local_data_[width_], local_height_ * width_, MPI_INT,
              world.rank() == 0 ? processed_data_.data() : nullptr, recvcounts.data(), displs.data(), MPI_INT, 0, comm);
}

const std::vector<int>& SimpleIntMPI::getDataPath() const { return data_path_; }

}  // namespace anufriev_d_linear_image