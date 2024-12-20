#include "mpi/konkov_i_linear_hist_stretch/include/ops_mpi.hpp"

#include <algorithm>

namespace konkov_i_linear_hist_stretch {

LinearHistogramStretch::LinearHistogramStretch(int image_size, int* image_data)
    : image_size_(image_size), image_data_(image_data), local_data_(nullptr) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  local_size_ = image_size_ / size_;
  if (rank_ < image_size_ % size_) {
    local_size_++;
  }

  local_data_ = new int[local_size_];
}

LinearHistogramStretch::~LinearHistogramStretch() { delete[] local_data_; }

bool LinearHistogramStretch::validation() const { return image_size_ > 0 && image_data_ != nullptr; }

bool LinearHistogramStretch::pre_processing() {
  if (!validation()) return false;

  distribute_data();
  return true;
}

bool LinearHistogramStretch::run() {
  // Compute local min and max
  int local_min = *std::min_element(local_data_, local_data_ + local_size_);
  int local_max = *std::max_element(local_data_, local_data_ + local_size_);

  // Compute global min and max
  int global_min, global_max;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  // Stretch local pixels
  if (global_max > global_min) {
    for (int i = 0; i < local_size_; ++i) {
      local_data_[i] =
          static_cast<int>((static_cast<double>(local_data_[i] - global_min) / (global_max - global_min)) * 255.0);
    }
  }
  return true;
}

bool LinearHistogramStretch::post_processing() {
  gather_data();
  return true;
}

void LinearHistogramStretch::distribute_data() {
  int* send_counts = new int[size_];
  int* displacements = new int[size_];
  displacements[0] = 0;

  for (int i = 0; i < size_; ++i) {
    send_counts[i] = image_size_ / size_;
    if (i < image_size_ % size_) {
      send_counts[i]++;
    }
    if (i > 0) {
      displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }
  }

  MPI_Scatterv(image_data_, send_counts, displacements, MPI_INT, local_data_, local_size_, MPI_INT, 0, MPI_COMM_WORLD);

  delete[] send_counts;
  delete[] displacements;
}

void LinearHistogramStretch::gather_data() {
  int* send_counts = new int[size_];
  int* displacements = new int[size_];
  displacements[0] = 0;

  for (int i = 0; i < size_; ++i) {
    send_counts[i] = image_size_ / size_;
    if (i < image_size_ % size_) {
      send_counts[i]++;
    }
    if (i > 0) {
      displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }
  }

  MPI_Gatherv(local_data_, local_size_, MPI_INT, image_data_, send_counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

  delete[] send_counts;
  delete[] displacements;
}

}  // namespace konkov_i_linear_hist_stretch
