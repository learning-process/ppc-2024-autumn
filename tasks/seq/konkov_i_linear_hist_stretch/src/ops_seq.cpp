#include "seq/konkov_i_linear_hist_stretch/include/ops_seq.hpp"

#include <algorithm>

namespace konkov_i_linear_hist_stretch {

LinearHistogramStretch::LinearHistogramStretch(int image_size, int* image_data)
    : image_size_(image_size), image_data_(image_data) {}

bool LinearHistogramStretch::validation() const { return image_size_ > 0 && image_data_ != nullptr; }

bool LinearHistogramStretch::pre_processing() {
  if (!validation()) return false;

  calculate_global_min_max();
  return true;
}

bool LinearHistogramStretch::run() {
  stretch_pixels();
  return true;
}

bool LinearHistogramStretch::post_processing() { return true; }

void LinearHistogramStretch::calculate_global_min_max() {
  global_min_ = *std::min_element(image_data_, image_data_ + image_size_);
  global_max_ = *std::max_element(image_data_, image_data_ + image_size_);
}

void LinearHistogramStretch::stretch_pixels() {
  if (global_max_ - global_min_ == 0) {
    return;
  }

  for (int i = 0; i < image_size_; ++i) {
    image_data_[i] =
        static_cast<int>((static_cast<double>(image_data_[i] - global_min_) / (global_max_ - global_min_)) * 255.0);
  }
}

}  // namespace konkov_i_linear_hist_stretch