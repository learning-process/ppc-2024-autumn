#include "mpi/mezhuev_m_sobel_edge_detection/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <iostream>
#include <vector>

namespace mezhuev_m_sobel_edge_detection {

bool SobelEdgeDetection::validation() {
  internal_order_test();
  if (!taskData || taskData->inputs.empty() || taskData->outputs.empty()) {
    return false;
  }
  if (taskData->inputs.size() != 1 || taskData->outputs.size() != 1) {
    return false;
  }
  if (taskData->inputs[0] == nullptr || taskData->outputs[0] == nullptr) {
    return false;
  }
  if (taskData->inputs_count.empty() || taskData->outputs_count.empty()) {
    return false;
  }
  if (taskData->inputs_count[0] != taskData->outputs_count[0]) {
    return false;
  }
  if (taskData->inputs_count[0] == 0 || taskData->outputs_count[0] == 0) {
    return false;
  }
  return true;
}

bool SobelEdgeDetection::pre_processing() {
  internal_order_test();
  if (taskData->inputs_count.empty() || taskData->inputs_count[0] == 0) {
    return false;
  }
  if (taskData->inputs.size() != 1) {
    return false;
  }
  size_t data_size = taskData->inputs_count[0];
  gradient_x.resize(data_size);
  gradient_y.resize(data_size);
  return true;
}

bool SobelEdgeDetection::run() {
  internal_order_test();

  uint8_t* input_image = taskData->inputs[0];
  uint8_t* output_image = taskData->outputs[0];
  size_t data_size = taskData->inputs_count[0];
  int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  int sobel_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
  int rank = world.rank();
  int size = world.size();
  size_t rows_per_process = data_size / size;
  size_t extra_rows = data_size % size;
  size_t start_row = rank * rows_per_process + std::min(rank, static_cast<int>(extra_rows));
  size_t end_row = (rank + 1) * rows_per_process + std::min(rank + 1, static_cast<int>(extra_rows));
  if (end_row > start_row + 1) {
    for (size_t y = start_row + 1; y < end_row - 1; ++y) {
      for (size_t x = 1; x < data_size - 1; ++x) {
        int gx = 0;
        int gy = 0;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int pixel_value = input_image[(y + ky) * data_size + (x + kx)];
            gx += sobel_x[ky + 1][kx + 1] * pixel_value;
            gy += sobel_y[ky + 1][kx + 1] * pixel_value;
          }
        }
        int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));
        output_image[y * data_size + x] = std::min(magnitude, 255);
      }
    }
  }
  size_t total_rows = data_size;
  if (rank == 0) {
    for (int i = 1; i < size; ++i) {
      size_t worker_start_row = i * rows_per_process + std::min(i, static_cast<int>(extra_rows));
      size_t worker_end_row = (i + 1) * rows_per_process + std::min(i + 1, static_cast<int>(extra_rows));
      if (worker_end_row > total_rows) {
        worker_end_row = total_rows;
      }
      world.recv(i, 0, output_image + worker_start_row * data_size, (worker_end_row - worker_start_row) * data_size);
    }
  } else {
    world.send(0, 0, output_image + start_row * data_size, (end_row - start_row) * data_size);
  }
  return true;
}

bool SobelEdgeDetection::post_processing() {
  internal_order_test();
  if (!taskData || taskData->outputs[0] == nullptr) {
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