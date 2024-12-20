#include "mpi/konkov_i_linear_hist_stretch/include/ops_mpi.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <climits>

namespace konkov_i_linear_hist_stretch {

LinearHistogramStretch::LinearHistogramStretch(int image_size, int* image_data)
    : image_size_(image_size), image_data_(image_data) {
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

  if (rank_ == 0) {
    int total_count = std::accumulate(send_counts, send_counts + size_, 0);
    if (total_count != image_size_) {
      std::cerr << "Error: Total send counts mismatch: " << total_count << " vs. " << image_size_ << std::endl;
      delete[] send_counts;
      delete[] displacements;
      return false;
    }
  }

  MPI_Scatterv(image_data_, send_counts, displacements, MPI_INT, local_data_, local_size_, MPI_INT, 0, MPI_COMM_WORLD);

  int local_min = local_size_ > 0 ? *std::min_element(local_data_, local_data_ + local_size_) : INT_MAX;
  int local_max = local_size_ > 0 ? *std::max_element(local_data_, local_data_ + local_size_) : INT_MIN;

  MPI_Allreduce(&local_min, &global_min_, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max_, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  delete[] send_counts;
  delete[] displacements;

  return true;
}

bool LinearHistogramStretch::run() {
  stretch_pixels();
  return true;
}

void LinearHistogramStretch::stretch_pixels() {
  if (global_max_ - global_min_ == 0) {
    return;
  }

  for (int i = 0; i < local_size_; ++i) {
    local_data_[i] =
        static_cast<int>((static_cast<double>(local_data_[i] - global_min_) / (global_max_ - global_min_)) * 255.0);
  }
}

bool LinearHistogramStretch::post_processing() {
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

  return true;
}

}  // namespace konkov_i_linear_hist_stretch
