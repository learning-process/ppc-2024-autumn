#include "seq/fomin_v_sobel_edges/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

bool fomin_v_sobel_edges::SobelEdgeDetection::pre_processing() {
  internal_order_test();

  width_ = taskData->inputs_count[0];
  height_ = taskData->inputs_count[1];
  input_image_.assign(reinterpret_cast<unsigned char*>(taskData->inputs[0]),
                      reinterpret_cast<unsigned char*>(taskData->inputs[0]) + width_ * height_);
  output_image_.resize(width_ * height_, 0);
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetection::validation() {
  internal_order_test();

  return taskData->inputs_count.size() == 2 && taskData->outputs_count.size() == 2;
}

bool fomin_v_sobel_edges::SobelEdgeDetection::run() {
  internal_order_test();

  const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  for (int y = 1; y < height_ - 1; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      int sumX = 0, sumY = 0;

      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          int pixel = input_image_[(y + i) * width_ + (x + j)];
          sumX += pixel * Gx[i + 1][j + 1];
          sumY += pixel * Gy[i + 1][j + 1];
        }
      }

      int gradient = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
      output_image_[y * width_ + x] = static_cast<unsigned char>(std::min(gradient, 255));
    }
  }
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetection::post_processing() {
  internal_order_test();

  std::copy(output_image_.begin(), output_image_.end(), reinterpret_cast<unsigned char*>(taskData->outputs[0]));
  return true;
}