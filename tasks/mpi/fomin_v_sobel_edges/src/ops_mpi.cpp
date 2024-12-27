#include "mpi/fomin_v_sobel_edges/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
using namespace std::chrono_literals;

#include <boost/mpi.hpp>

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::pre_processing() {
  internal_order_test();

  width_ = taskData->inputs_count[0];
  height_ = taskData->inputs_count[1];
  input_image_.assign(taskData->inputs[0], taskData->inputs[0] + width_ * height_);
  output_image_.resize(width_ * height_, 0);
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::validation() {
  internal_order_test();
  return taskData->inputs_count.size() == 2 && taskData->outputs_count.size() == 2;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::run() {
  internal_order_test();
  boost::mpi::communicator world;

  int rank = world.rank();
  int size = world.size();

  const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  int rows_per_process = (height_ - 2) / size;
  int start_row = rank * rows_per_process + 1;
  int end_row = (rank == size - 1) ? height_ - 1 : start_row + rows_per_process;

  std::vector<unsigned char> top_row(width_);
  std::vector<unsigned char> bottom_row(width_);

  if (rank > 0) {
    std::vector<unsigned char> send_top_row(input_image_.begin() + start_row * width_,
                                            input_image_.begin() + start_row * width_ + width_);
    world.send(rank - 1, 0, send_top_row);
    world.recv(rank - 1, 1, top_row);
  }

  if (rank < size - 1) {
    std::vector<unsigned char> send_bottom_row(input_image_.begin() + (end_row - 1) * width_,
                                               input_image_.begin() + (end_row)*width_);
    world.send(rank + 1, 1, send_bottom_row);
    world.recv(rank + 1, 0, bottom_row);
  }

  for (int y = start_row; y < end_row; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      int sumX = 0;
      int sumY = 0;

      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          int pixel_row = y + i;
          if (pixel_row < start_row - 1) pixel_row = start_row - 1;
          if (pixel_row > end_row) pixel_row = end_row;

          unsigned char pixel;
          if (pixel_row == start_row - 1 && rank > 0)
            pixel = top_row[x + j];
          else if (pixel_row == end_row && rank < size - 1)
            pixel = bottom_row[x + j];
          else
            pixel = input_image_[pixel_row * width_ + (x + j)];

          sumX += pixel * Gx[i + 1][j + 1];
          sumY += pixel * Gy[i + 1][j + 1];
        }
      }

      int gradient = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
      output_image_[y * width_ + x] = static_cast<unsigned char>(std::min(gradient, 255));
    }
  }

  if (rank != 0) {
    std::vector<unsigned char> send_buffer(output_image_.begin() + start_row * width_,
                                           output_image_.begin() + end_row * width_);
    world.send(0, 2, send_buffer);
  } else {
    for (int i = 1; i < size; ++i) {
      int proc_start_row = i * rows_per_process + 1;
      std::vector<unsigned char> recv_buffer;
      world.recv(i, 2, recv_buffer);
      std::copy(recv_buffer.begin(), recv_buffer.end(), output_image_.begin() + proc_start_row * width_);
    }
  }

  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::post_processing() {
  internal_order_test();

  if (taskData->outputs[0] != nullptr) {
    std::copy(output_image_.begin(), output_image_.end(), taskData->outputs[0]);
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