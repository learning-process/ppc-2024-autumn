#include "mpi/anufriev_d_linear_image/include/ops_mpi_anufriev.hpp"
#include <gtest/gtest.h>
#include <boost/mpi.hpp>
#include <numeric>
#include <algorithm>
#include <mpi.h>

namespace anufriev_d_linear_image {

SimpleIntMPI::SimpleIntMPI(const std::shared_ptr<ppc::core::TaskData>& taskData) : Task(taskData) {}

bool SimpleIntMPI::pre_processing() {
  internal_order_test();
  return true;
}

bool SimpleIntMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (!taskData || taskData->inputs.size() < 3 || taskData->inputs_count.size() < 3 ||
        taskData->outputs.empty() || taskData->outputs_count.empty()) {
      return false;
    }

    width_ = *reinterpret_cast<int*>(taskData->inputs[1]);
    height_ = *reinterpret_cast<int*>(taskData->inputs[2]);

    size_t expected_size = static_cast<size_t>(width_ * height_);
    if (width_ < 3 || height_ < 3 ||
        taskData->inputs_count[0] != expected_size ||
        taskData->outputs_count[0] != expected_size) {
      return false;
    }

    // Перестраиваем данные в column-major порядок
    original_data_.resize(width_ * height_);
    int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for(int c = 0; c < width_; c++) {
      for(int r = 0; r < height_; r++) {
        original_data_[c * height_ + r] = input_ptr[r * width_ + c];
      }
    }
  }

  // Распространяем width и height на все процессы
  boost::mpi::broadcast(world, width_, 0);
  boost::mpi::broadcast(world, height_, 0);
  total_size_ = static_cast<size_t>(width_ * height_);

  return true;
}

bool SimpleIntMPI::run() {
  internal_order_test();

  distributeData();
  applyGaussianFilter();
  gatherData();

  return true;
}

bool SimpleIntMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Перестраиваем данные обратно в row-major порядок
    int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    for(int r = 0; r < height_; r++) {
      for(int c = 0; c < width_; c++) {
        output_ptr[r * width_ + c] = processed_data_[c * height_ + r];
      }
    }
  }
  return true;
}

void SimpleIntMPI::distributeData() {
  MPI_Comm comm = world;
  int nprocs = world.size();

  std::vector<int> sendcounts(nprocs, 0);
  std::vector<int> displs(nprocs, 0);
  int base_cols = width_ / nprocs;
  int remainder = width_ % nprocs;

  int offset = 0;
  for (int i = 0; i < nprocs; ++i) {
    int part_cols = base_cols + (i < remainder ? 1 : 0);
    sendcounts[i] = part_cols * height_;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  local_width_ = base_cols + (world.rank() < remainder ? 1 : 0);
  start_col_ = displs[world.rank()] / height_;

  local_data_.resize(sendcounts[world.rank()]);

  MPI_Scatterv(world.rank() == 0 ? original_data_.data() : nullptr,
               sendcounts.data(), displs.data(), MPI_INT,
               local_data_.data(), sendcounts[world.rank()], MPI_INT,
               0, comm);
}

void SimpleIntMPI::gatherData() {
  MPI_Comm comm = world;
  int nprocs = world.size();

  std::vector<int> recvcounts(nprocs, 0);
  std::vector<int> displs(nprocs, 0);
  int base_cols = width_ / nprocs;
  int remainder = width_ % nprocs;

  int offset = 0;
  for (int i = 0; i < nprocs; ++i) {
    int part_cols = base_cols + (i < remainder ? 1 : 0);
    recvcounts[i] = part_cols * height_;
    displs[i] = offset;
    offset += recvcounts[i];
  }

  if (world.rank() == 0) {
    processed_data_.resize(width_ * height_);
  }

  MPI_Gatherv(local_data_.data(), recvcounts[world.rank()], MPI_INT,
              world.rank() == 0 ? processed_data_.data() : nullptr,
              recvcounts.data(), displs.data(), MPI_INT,
              0, comm);
}

void SimpleIntMPI::exchangeHalo(std::vector<int>& local_data, int local_width) {
  MPI_Comm comm = world;
  int left_rank = (world.rank() > 0) ? world.rank() - 1 : MPI_PROC_NULL;
  int right_rank = (world.rank() < world.size() - 1) ? world.rank() + 1 : MPI_PROC_NULL;

  std::vector<int> left_col(height_), right_col(height_);
  std::vector<int> send_left(height_), send_right(height_);
  for (int r = 0; r < height_; ++r) {
    send_left[r] = local_data[r * local_width];                  // левый край локальных данных
    send_right[r] = local_data[r * local_width + local_width - 1]; // правый край локальных данных
  }

  MPI_Request reqs[4];
  int req_count = 0;

  // Передача левого края налево, получение оттуда правого края
  if (left_rank != MPI_PROC_NULL) {
    MPI_Isend(send_left.data(), height_, MPI_INT, left_rank, 0, comm, &reqs[req_count++]);
    MPI_Irecv(left_col.data(), height_, MPI_INT, left_rank, 1, comm, &reqs[req_count++]);
  }

  // Передача правого края направо, получение слева
  if (right_rank != MPI_PROC_NULL) {
    MPI_Isend(send_right.data(), height_, MPI_INT, right_rank, 1, comm, &reqs[req_count++]);
    MPI_Irecv(right_col.data(), height_, MPI_INT, right_rank, 0, comm, &reqs[req_count++]);
  }

  MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

  int extended_width = local_width + 2;
  std::vector<int> extended_data(extended_width * height_);

  for (int r = 0; r < height_; ++r) {
    // Гало слева
    extended_data[r * extended_width + 0] = (left_rank != MPI_PROC_NULL)
        ? left_col[r]
        : local_data[r * local_width]; // дублируем левый пиксель если соседа нет

    // Основная часть
    for (int c = 0; c < local_width; ++c) {
      extended_data[r * extended_width + (c + 1)] = local_data[r * local_width + c];
    }

    // Гало справа
    extended_data[r * extended_width + (local_width + 1)] = (right_rank != MPI_PROC_NULL)
        ? right_col[r]
        : local_data[r * local_width + local_width - 1]; // дублируем правый пиксель если соседа нет
  }

  local_data.swap(extended_data);
}

void SimpleIntMPI::applyGaussianFilter() {
  if (local_width_ <= 0) return;

  std::vector<int> extended_data = local_data_;
  exchangeHalo(extended_data, local_width_);

  int extended_width = local_width_ + 2;
  std::vector<int> result(height_ * local_width_, 0);

  for (int r = 0; r < height_; ++r) {
    for (int c = 0; c < local_width_; ++c) {
      int sum = 0;
      for (int kr = -1; kr <= 1; ++kr) {
        for (int kc = -1; kc <= 1; ++kc) {
          int rr = r + kr;
          int cc = (c + 1) + kc; // +1 из-за смещения основной области внутри расширенной

          rr = std::max(0, std::min(rr, height_ - 1));
          cc = std::max(0, std::min(cc, extended_width - 1));

          sum += extended_data[rr * extended_width + cc] * kernel_[kr + 1][kc + 1];
        }
      }
      result[r * local_width_ + c] = sum / 16;
    }
  }

  local_data_.swap(result);
}

const std::vector<int>& SimpleIntMPI::getDataPath() const {
  return data_path_;
}

}  // namespace anufriev_d_linear_image