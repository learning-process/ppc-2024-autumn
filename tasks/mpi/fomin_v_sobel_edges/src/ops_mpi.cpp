#include "mpi/fomin_v_sobel_edges/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
using namespace std::chrono_literals;

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_image_ = *reinterpret_cast<std::vector<unsigned char>*>(taskData->inputs[0]);
    width_ = taskData->inputs_count[0];
    height_ = taskData->inputs_count[1];
  }

  broadcast(world, width_, 0);
  broadcast(world, height_, 0);

  int delta_height = height_ / world.size();
  local_height_ = delta_height;
  if (world.rank() == world.size() - 1) {
    local_height_ = height_ - delta_height * (world.size() - 1);
  }

  local_input_image_.resize((local_height_ + 2) * width_, 0);
  local_output_image_.resize(local_height_ * width_, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); ++proc) {
      int start_row = proc * delta_height;
      int end_row = (proc == world.size() - 1) ? height_ : (proc + 1) * delta_height;
      int rows_to_send = end_row - start_row;

      world.send(proc, 0, input_image_.data() + start_row * width_, rows_to_send * width_);
    }
    std::copy(input_image_.begin(), input_image_.begin() + (delta_height + 1) * width_, local_input_image_.begin());
  } else {
    world.recv(0, 0, local_input_image_.data() + width_, local_height_ * width_);
  }

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

  for (int y = 1; y < local_height_ + 1; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      int sumX = 0, sumY = 0;

      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          int pixel = local_input_image_[(y + i) * width_ + (x + j)];
          sumX += pixel * Gx[i + 1][j + 1];
          sumY += pixel * Gy[i + 1][j + 1];
        }
      }

      int gradient = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
      local_output_image_[(y - 1) * width_ + x] = static_cast<unsigned char>(std::min(gradient, 255));
    }
  }

  if (world.rank() > 0) {
    world.send(world.rank() - 1, 0, local_input_image_.data() + width_, width_);

    world.recv(world.rank() - 1, 0, local_input_image_.data(), width_);
  }
  if (world.rank() < world.size() - 1) {
    world.send(world.rank() + 1, 0, local_input_image_.data() + local_height_ * width_, width_);
    world.recv(world.rank() + 1, 0, local_input_image_.data() + (local_height_ + 1) * width_, width_);
  }

  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    output_image_.resize(width_ * height_, 0);
    std::copy(local_output_image_.begin(), local_output_image_.end(), output_image_.begin());
  }

  if (world.rank() > 0) {
    world.send(0, 0, local_output_image_.data(), local_height_ * width_);
  } else {
    for (int proc = 1; proc < world.size(); ++proc) {
      int start_row = proc * (height_ / world.size());
      int end_row = (proc == world.size() - 1) ? height_ : (proc + 1) * (height_ / world.size());
      int rows_to_recv = end_row - start_row;

      world.recv(proc, 0, output_image_.data() + start_row * width_, rows_to_recv * width_);
    }
  }

  if (world.rank() == 0) {
    *reinterpret_cast<std::vector<unsigned char>*>(taskData->outputs[0]) = output_image_;
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

  *reinterpret_cast<std::vector<unsigned char>*>(taskData->outputs[0]) = output_image_;
  return true;
}