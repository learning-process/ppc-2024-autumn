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

  int num_procs = world.size();
  int local_height = (height_ - 2) / num_procs;
  int reminder = (height_ - 2) % num_procs;

  std::vector<unsigned char> local_input;

  if (world.rank() == 0) {
    // Prepare and send data to other processes
    for (int proc = 1; proc < num_procs; ++proc) {
      int start_row = reminder > 0 ? 1 + proc : 1 + local_height * proc;
      int proc_local_height = local_height + (proc <= reminder ? 1 : 0);
      local_input.resize((proc_local_height + 2) * width_);  // Include boundary rows

      for (int y = start_row - 1; y < start_row + proc_local_height + 1; ++y) {
        for (int x = 0; x < width_; ++x) {
          local_input[(y - (start_row - 1)) * width_ + x] = input_image_[y * width_ + x];
        }
      }
      world.send(proc, 0, local_input.data(), local_input.size());
    }

    // Prepare local input for root process
    local_height = reminder + local_height;
    local_input.resize((local_height + 2) * width_);
    for (int y = 0; y < local_height + 2; ++y) {
      for (int x = 0; x < width_; ++x) {
        local_input[y * width_ + x] = input_image_[y * width_ + x];
      }
    }
  } else {
    // Receive data from root process
    world.recv(0, 0, local_input.data(), local_input.size());
  }

  int local_height_effective = local_input.size() / width_ - 2;  // Exclude boundary rows
  std::vector<unsigned char> local_output(local_height_effective * width_, 0);

  for (int y = 1; y < local_height_effective + 1; ++y) {
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
      local_output[(y - 1) * width_ + (x - 1)] = static_cast<unsigned char>(std::min(gradient, 255));
    }
  }

  std::vector<int> local_sizes(num_procs, local_height * width_);
  if (world.rank() == 0) {
    for (int proc = 1; proc < num_procs; ++proc) {
      local_sizes[proc] = (local_height + (proc <= reminder ? 1 : 0)) * width_;
    }
  }

  // Gather results
  std::vector<unsigned char> gathered_output((height_ - 2) * width_, 0);
  if (world.rank() == 0) {
    gathered_output.resize((height_ - 2) * width_);
  }

  gatherv(world, local_output.data(), local_output.size(), gathered_output.data(), local_sizes, 0);

  if (world.rank() == 0) {
    output_image_.assign(gathered_output.begin(), gathered_output.end());
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