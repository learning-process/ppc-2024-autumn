// Copyright 2024 Nesterov Alexander
#include "seq/zolotareva_a_smoothing_image/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool zolotareva_a_smoothing_image_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0;
}

bool zolotareva_a_smoothing_image_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  height_ = taskData->inputs_count[0];
  width_ = taskData->inputs_count[1];
  input_.resize(height_ * width_);
  const uint8_t* raw_data = reinterpret_cast<uint8_t*>(taskData->inputs[0]);

  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      input_[i * width_ + j] = raw_data[i * width_ + j];
    }
  }
  result_.resize(height_ * width_);
  return true;
}

bool zolotareva_a_smoothing_image_seq::TestTaskSequential::run() {
  internal_order_test();
  int radius = 1;
  float sigma = 1.0f;
  std::vector<float> horizontal_kernel = create_gaussian_kernel(radius, sigma);
  std::vector<float>& vertical_kernel = horizontal_kernel;
  std::vector<float> temp(height_ * width_, 0.0f);

  convolve_rows(input_, height_, width_, horizontal_kernel, temp);
  convolve_columns(temp, height_, width_, vertical_kernel, result_);
  return true;
}

bool zolotareva_a_smoothing_image_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* output_raw = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
  for (int i = 0; i < height_; i++) {
    for (int j = 0; j < width_; j++) {
      output_raw[i * width_ + j] = result_[i * width_ + j];
    }
  }
  return true;
}
std::vector<float> zolotareva_a_smoothing_image_seq::TestTaskSequential::create_gaussian_kernel(int radius,
                                                                                                float sigma) {
  int size = 2 * radius + 1;
  std::vector<float> kernel(size);
  float norm = 0.0f;
  for (int i = -radius; i <= radius; ++i) {
    kernel[i + radius] = std::exp(-(i * i) / (2 * sigma * sigma));
    norm += kernel[i + radius];
  }
  for (float& val : kernel) {
    val /= norm;
  }
  return kernel;
}
void zolotareva_a_smoothing_image_seq::TestTaskSequential::convolve_rows(const std::vector<uint8_t>& input, int height,
                                                                         int width, const std::vector<float>& kernel,
                                                                         std::vector<float>& temp) {
  int kernel_radius = kernel.size() / 2;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -kernel_radius; k <= kernel_radius; ++k) {
        int pixel_x = std::clamp(x + k, 0, width - 1);
        sum += input[y * width + pixel_x] * kernel[k + kernel_radius];
      }
      temp[y * width + x] = sum;
    }
  }
}
void zolotareva_a_smoothing_image_seq::TestTaskSequential::convolve_columns(const std::vector<float>& temp, int height,
                                                                            int width, const std::vector<float>& kernel,
                                                                            std::vector<uint8_t>& output) {
  int kernel_radius = kernel.size() / 2;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -kernel_radius; k <= kernel_radius; ++k) {
        int pixel_y = std::clamp(y + k, 0, height - 1);
        sum += temp[pixel_y * width + x] * kernel[k + kernel_radius];
      }
      output[y * width + x] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(sum)), 0, 255));
    }
  }
}