#include "seq/mezhuev_m_sobel_edge_detection/include/seq.hpp"

#include <cmath>
#include <iostream>

namespace mezhuev_m_sobel_edge_detection {

bool SobelEdgeDetectionSeq::validation() {
  if (taskData == nullptr) {
    std::cerr << "Error: taskData is nullptr." << std::endl;
    return false;
  }

  if (taskData->inputs.empty() || taskData->outputs.empty()) {
    std::cerr << "Error: taskData buffers are null or empty." << std::endl;
    return false;
  }

  if (taskData->inputs.size() != 1 || taskData->outputs.size() != 1) {
    std::cerr << "Error: Expected exactly one input and one output." << std::endl;
    return false;
  }

  if (taskData->inputs[0] == nullptr || taskData->outputs[0] == nullptr) {
    std::cerr << "Error: Invalid input or output buffer." << std::endl;
    return false;
  }

  size_t input_size = taskData->width * taskData->height;
  size_t output_size = taskData->width * taskData->height;

  if (taskData->inputs_count[0] != input_size || taskData->outputs_count[0] != output_size) {
    std::cerr << "Error: Mismatch in input/output buffer sizes." << std::endl;
    return false;
  }

  return true;
}

bool SobelEdgeDetectionSeq::pre_processing(TaskData* task_data) {
  if (task_data == nullptr) {
    std::cerr << "Error: task_data is nullptr before processing!" << std::endl;
    return false;  // Возвращаем false, если task_data равен nullptr
  }

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
    std::cerr << "Error: task_data is nullptr in run." << std::endl;
    return false;
  }
  std::cout << "Running Sobel edge detection..." << std::endl;
  std::cout << "Task data is valid: "
            << "Width: " << taskData->width << ", Height: " << taskData->height
            << ", Inputs: " << taskData->inputs.size() << ", Outputs: " << taskData->outputs.size() << std::endl;
  size_t width = taskData->width;
  size_t height = taskData->height;
  uint8_t* input_image = taskData->inputs[0];
  uint8_t* output_image = taskData->outputs[0];

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
      magnitude = std::min(magnitude, 255);
      output_image[y * width + x] = static_cast<uint8_t>(magnitude);
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