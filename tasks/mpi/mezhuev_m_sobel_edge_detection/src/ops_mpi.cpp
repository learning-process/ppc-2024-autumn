#include "mpi/mezhuev_m_sobel_edge_detection/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <iostream>
#include <vector>

namespace mezhuev_m_sobel_edge_detection {

bool SobelEdgeDetectionMPI::validation() {
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

bool SobelEdgeDetectionMPI::pre_processing(TaskData* task_data) {
  if (!validation()) {
    return false;
  }

  taskData = task_data;
  gradient_x.resize(taskData->width * taskData->height);
  gradient_y.resize(taskData->width * taskData->height);

  taskData->outputs[0] = new uint8_t[taskData->width * taskData->height]();

  return true;
}

bool SobelEdgeDetectionMPI::run() {
  if (taskData == nullptr) {
    return false;
  }

  int rank = world.rank();
  int size = world.size();
  size_t width = taskData->width;
  size_t height = taskData->height;

  uint8_t* input_image = taskData->inputs[0];
  uint8_t* output_image = taskData->outputs[0];

  int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  int sobel_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

  size_t rows_per_process = height / size;
  size_t extra_rows = height % size;

  size_t start_row = static_cast<size_t>(rank * rows_per_process + std::min(rank, static_cast<int>(extra_rows)));
  size_t end_row =
      static_cast<size_t>((rank + 1) * rows_per_process + std::min(rank + 1, static_cast<int>(extra_rows)));

  for (size_t y = start_row + 1; y < end_row - 1; ++y) {
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

  world.barrier();

  return true;
}

bool SobelEdgeDetectionMPI::post_processing() {
  if (taskData == nullptr || taskData->outputs[0] == nullptr) {
    return false;
  }

  for (size_t i = 0; i < taskData->outputs_count[0]; ++i) {
    if (taskData->outputs[0][i] != 0) {
      return true;
    }
  }

  return false;
}

}  // namespace mezhuev_m_sobel_edge_detection