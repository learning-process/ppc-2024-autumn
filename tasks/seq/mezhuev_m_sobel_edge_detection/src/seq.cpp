#include "seq/mezhuev_m_sobel_edge_detection/include/seq.hpp"

#include <cmath>
#include <iostream>

namespace mezhuev_m_sobel_edge_detection {

bool SobelEdgeDetectionSeq::validation() {
  if (taskData == nullptr || taskData->inputs.empty() || taskData->outputs.empty()) {
    return false;
  }

  if (taskData->inputs.size() != 1 || taskData->outputs.size() != 1) {
    return false;
  }

  if (taskData->inputs[0] == nullptr || taskData->outputs[0] == nullptr) {
    return false;
  }

  if (taskData->inputs_count.empty() || taskData->outputs_count.empty() ||
      taskData->inputs_count[0] != taskData->outputs_count[0]) {
    return false;
  }

  return true;
}

bool SobelEdgeDetectionSeq::pre_processing(TaskData* task_data) {
  if (!validation()) {
    return false;
  }

  gradient_x.resize(task_data->width * task_data->height);
  gradient_y.resize(task_data->width * task_data->height);

  taskData = task_data;
  return true;
}

bool SobelEdgeDetectionSeq::run() {
  if (taskData == nullptr) {
    return false;
  }

  size_t width = taskData->width;
  size_t height = taskData->height;
  uint8_t* input_image = taskData->inputs[0];
  uint8_t* output_image = taskData->outputs[0];

  gradient_x.resize(width * height, 0);
  gradient_y.resize(width * height, 0);

  int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  int sobel_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

  for (size_t y = 1; y < height - 1; ++y) {
    for (size_t x = 1; x < width - 1; ++x) {
      int gx = 0;
      int gy = 0;

      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          int pixel_value = input_image[(y + ky) * width + (x + kx)];
          gx += sobel_x[ky + 1][kx + 1] * pixel_value;
          gy += sobel_y[ky + 1][kx + 1] * pixel_value;
        }
      }

      int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));
      output_image[y * width + x] = std::min(magnitude, 255);
    }
  }

  return true;
}

bool SobelEdgeDetectionSeq::post_processing() {
  if (taskData == nullptr || taskData->outputs[0] == nullptr) {
    std::cerr << "Error: Invalid output buffer." << std::endl;
    return false;
  }

  for (size_t i = 0; i < taskData->outputs_count[0]; ++i) {
    if (taskData->outputs[0][i] == 0) {
      std::cerr << "Error: Invalid output value at index " << i << "." << std::endl;
      return false;
    }
  }

  return true;
}

}  // namespace mezhuev_m_sobel_edge_detection