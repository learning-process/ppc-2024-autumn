#include "../include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>

int16_t conv_kernel3(const std::array<std::array<int8_t, 3>, 3>& kernel, const std::vector<uint8_t>& img, size_t i,
                     size_t j, size_t height) {
  // clang-format off
  return kernel[0][0] * img[(i - 1) * height + (j - 1)]
       + kernel[0][1] * img[(i - 1) * height + j]
       + kernel[0][2] * img[(i - 1) * height + (j + 1)]
       + kernel[1][0] * img[i * height + (j - 1)]
       + kernel[1][1] * img[i * height + j]
       + kernel[1][2] * img[i * height + (j + 1)]
       + kernel[2][0] * img[(i + 1) * height + (j - 1)]
       + kernel[2][1] * img[(i + 1) * height + j]
       + kernel[2][2] * img[(i + 1) * height + (j + 1)];
  // clang-format on
}

bool koshkin_m_sobel_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  imgsize = {taskData->inputs_count[0], taskData->inputs_count[1]};
  auto& [width, height] = imgsize;
  const int padding = 2;
  image.resize((width + padding) * (height + padding));
  resimg.resize(width * height, 0);

  const auto* in = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
  for (size_t row = 0; row < height; row++) {
    std::copy(in + (row * width), in + ((row + 1) * width), image.begin() + ((row + 1) * (width + padding) + 1));
  }

  return true;
}

bool koshkin_m_sobel_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->outputs_count[0] == taskData->inputs_count[0] &&
         taskData->outputs_count[1] == taskData->inputs_count[1];
}

bool koshkin_m_sobel_seq::TestTaskSequential::run() {
  internal_order_test();

  const auto [width, height] = imgsize;

  for (size_t i = 1; i < (width + 2) - 1; i++) {
    for (size_t j = 1; j < (height + 2) - 1; j++) {
      const auto accX = conv_kernel3(SOBEL_KERNEL_X, image, i, j, (height + 2));
      const auto accY = conv_kernel3(SOBEL_KERNEL_Y, image, i, j, (height + 2));
      resimg[(i - 1) * height + (j - 1)] = std::clamp(std::sqrt(std::pow(accX, 2) + std::pow(accY, 2)), 0., 255.);
    }
  }

  return true;
}

bool koshkin_m_sobel_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  std::copy(resimg.begin(), resimg.end(), reinterpret_cast<uint8_t*>(taskData->outputs[0]));
  return true;
}
