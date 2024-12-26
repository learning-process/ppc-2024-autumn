#include "mpi/fomin_v_sobel_edges/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
using namespace std::chrono_literals;

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    width_ = taskData->inputs_count[0];
    height_ = taskData->inputs_count[1];
    input_image_.assign(reinterpret_cast<unsigned char*>(taskData->inputs[0]),
                        reinterpret_cast<unsigned char*>(taskData->inputs[0]) + width_ * height_);
    output_image_.resize(width_ * height_, 0);
  }

  broadcast(world, width_, 0);
  broadcast(world, height_, 0);

  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return taskData->inputs_count.size() == 2 && taskData->outputs_count.size() == 2;
  }
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::run() {
  internal_order_test();

  const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  unsigned int blockSize = (height_ - 2) / world.size();
  unsigned int reminder = (height_ - 2) % world.size();

  std::vector<unsigned char> local_input((blockSize + 2) * width_);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      int start_row = proc * blockSize + reminder;
      world.send(proc, 0, input_image_.data() + start_row * width_, (blockSize + 2) * width_);
    }
    local_input.assign(input_image_.begin(), input_image_.begin() + (blockSize + reminder + 2) * width_);
  } else {
    world.recv(0, 0, local_input.data(), (blockSize + 2) * width_);
  }

  int local_height = local_input.size() / width_;
  std::vector<unsigned char> local_output(local_height * width_, 0);

  for (int y = 1; y < local_height - 1; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      int sumX = 0;
      int sumY = 0;

      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          int pixel = local_input[(y + i) * width_ + (x + j)];
          sumX += pixel * Gx[i + 1][j + 1];
          sumY += pixel * Gy[i + 1][j + 1];
        }
      }

      int gradient = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
      local_output[y * width_ + x] = static_cast<unsigned char>(std::min(gradient, 255));
    }
  }

  local_output.erase(local_output.begin(), local_output.begin() + width_);
  local_output.erase(local_output.end() - width_, local_output.end());

  std::vector<int> local_sizes(world.size(), blockSize * width_);
  local_sizes[0] += reminder * width_;

  gatherv(world, local_output.data(), local_output.size(), output_image_.data(), local_sizes, 0);

  if (world.rank() == 0) {
    std::vector<unsigned char> zero_row(width_, 0);
    output_image_.insert(output_image_.begin(), zero_row.begin(), zero_row.end());
    output_image_.erase(output_image_.end() - width_, output_image_.end());
  }

  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(output_image_.begin(), output_image_.end(), reinterpret_cast<unsigned char*>(taskData->outputs[0]));
  }

  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetection::pre_processing() {
  internal_order_test();

  input_image_ = *reinterpret_cast<std::vector<unsigned char>*>(taskData->inputs[0]);
  width_ = taskData->inputs_count[0];
  height_ = taskData->inputs_count[1];
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
      int sumX = 0;
      int sumY = 0;

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

  *reinterpret_cast<std::vector<unsigned char>*>(taskData->outputs[0]) = output_image_;
  return true;
}