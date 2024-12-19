#include "mpi/konkov_i_linear_hist_stretch/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>

namespace konkov_i_linear_hist_stretch {

LinearHistogramStretch::LinearHistogramStretch(int image_size, int* image_data)
    : image_size_(image_size), image_data_(image_data) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  local_size_ = image_size_ / size_;
  local_data_ = new int[local_size_];
}

bool LinearHistogramStretch::validation() const { return image_size_ > 0 && image_data_ != nullptr; }

bool LinearHistogramStretch::pre_processing() {
  if (!validation()) return false;

  MPI_Scatter(image_data_, local_size_, MPI_INT, local_data_, local_size_, MPI_INT, 0, MPI_COMM_WORLD);

  int local_min = *std::min_element(local_data_, local_data_ + local_size_);
  int local_max = *std::max_element(local_data_, local_data_ + local_size_);
  MPI_Allreduce(&local_min, &global_min_, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max_, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  return true;
}

bool LinearHistogramStretch::run() {
  stretch_pixels();
  return true;
}

bool LinearHistogramStretch::post_processing() {
  int* stretched_image = nullptr;
  if (rank_ == 0) {
    stretched_image = new int[image_size_];
  }
  MPI_Gather(local_data_, local_size_, MPI_INT, stretched_image, local_size_, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    for (int i = 0; i < image_size_; ++i) {
      image_data_[i] = stretched_image[i];
    }
    delete[] stretched_image;
  }
  delete[] local_data_;
  return true;
}

void LinearHistogramStretch::stretch_pixels() {
  for (int i = 0; i < local_size_; ++i) {
    if (global_max_ - global_min_ == 0) continue;
    local_data_[i] =
        static_cast<int>((static_cast<double>(local_data_[i] - global_min_) / (global_max_ - global_min_)) * 255.0);
  }
}

}  // namespace konkov_i_linear_hist_stretch