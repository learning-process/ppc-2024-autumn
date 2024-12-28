#include "mpi/fomin_v_sobel_edges/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
using namespace std::chrono_literals;

#include <boost/mpi.hpp>

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    width_ = taskData->inputs_count[0];
    height_ = taskData->inputs_count[1];
    input_image_.resize(width_ * height_);
    std::copy(reinterpret_cast<unsigned char*>(taskData->inputs[0]),
              reinterpret_cast<unsigned char*>(taskData->inputs[0]) + width_ * height_, input_image_.begin());
  }

  int local_height;
  if (world.rank() == 0) {
    int base_height = height_ / world.size();
    int extra = height_ % world.size();
    local_height = base_height + (world.rank() < extra ? 1 : 0);
  }

  // Broadcast width and height to all processes
  MPI_Bcast(&width_, 1, MPI_INT, 0, world);
  MPI_Bcast(&height_, 1, MPI_INT, 0, world);

  // Calculate local_height for each process
  int base_height = height_ / world.size();
  int extra = height_ % world.size();
  local_height = base_height + (world.rank() < extra ? 1 : 0);

  // Resize local_input_ with padding
  local_input_.resize((local_height + 2) * width_, 0);  // +2 for boundary rows

  if (world.rank() == 0) {
    // Prepare and send data to each process
    for (int proc = 0; proc < world.size(); ++proc) {
      int proc_local_height = base_height + (proc < extra ? 1 : 0);
      int start_row = base_height * proc + std::min(proc, extra);
      int end_row = start_row + proc_local_height;

      std::vector<unsigned char> send_data((proc_local_height + 2) * width_, 0);

      if (proc == 0) {
        // Handle rank 0's data locally
        // Top padding is the first row
        std::copy(input_image_.begin(), input_image_.begin() + width_, local_input_.begin());
        // Main data
        std::copy(input_image_.begin() + start_row * width_, input_image_.begin() + end_row * width_,
                  local_input_.begin() + width_);
        // Bottom padding is the next block's first row
        if (proc + 1 < world.size()) {
          int next_start_row = base_height * (proc + 1) + std::min(proc + 1, extra);
          std::copy(input_image_.begin() + next_start_row * width_,
                    input_image_.begin() + (next_start_row + 1) * width_,
                    local_input_.begin() + (proc_local_height + 1) * width_);
        } else {
          std::copy(input_image_.begin() + (height_ - 1) * width_, input_image_.begin() + height_ * width_,
                    local_input_.begin() + (proc_local_height + 1) * width_);
        }
      } else {
        // Prepare send_data for other processes
        int prev_end_row = base_height * proc + std::min(proc, extra);
        // Top padding is the previous block's last row
        std::copy(input_image_.begin() + (prev_end_row - 1) * width_, input_image_.begin() + prev_end_row * width_,
                  send_data.begin());
        // Main data
        std::copy(input_image_.begin() + start_row * width_, input_image_.begin() + end_row * width_,
                  send_data.begin() + width_);
        // Bottom padding is the next block's first row or the last row
        if (proc + 1 < world.size()) {
          int next_start_row = base_height * (proc + 1) + std::min(proc + 1, extra);
          std::copy(input_image_.begin() + next_start_row * width_,
                    input_image_.begin() + (next_start_row + 1) * width_,
                    send_data.begin() + (proc_local_height + 1) * width_);
        } else {
          std::copy(input_image_.begin() + (height_ - 1) * width_, input_image_.begin() + height_ * width_,
                     send_data.begin() + (proc_local_height + 1) * width_);
        }
        // Send data to process
        world.send(proc, 0, send_data.data(), send_data.size());
      }
    }
  } else {
    // Receive data from rank 0
    world.recv(0, 0, local_input_.data(), local_input_.size());
  }

  return true;
}
bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return taskData->inputs_count.size() == 2 && taskData->outputs_count.size() == 2;
  }
  if (world.rank() == 0) {
    std::cout << "Rank 0: inputs_count size = " << taskData->inputs_count.size()
              << ", outputs_count size = " << taskData->outputs_count.size() << std::endl;
  }
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::run() {
  internal_order_test();

  const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  int local_height = local_input_.size() / width_ - 2;  // exclude boundary rows
  output_image_.resize(local_height * width_, 0);       // Ensure output_image_ has correct size

  for (int y = 0; y < local_height; ++y) {  // Process all rows
    for (int x = 0; x < width_; ++x) {      // Process all columns
      int sumX = 0;
      int sumY = 0;
      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          int pixel = local_input_[(y + i + 1) * width_ + (x + j + 1)];  // Access with padding
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

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::post_processing() {
  internal_order_test();

  std::vector<int> sendcounts(world.size());
  std::vector<int> displacements(world.size());
  std::vector<unsigned char> gathered_output;

  gathered_output.resize(height_ * width_, 0);

  int base_height = height_ / world.size();
  int extra = height_ % world.size();
  for (int proc = 0; proc < world.size(); ++proc) {
    int proc_local_height = base_height + (proc < extra ? 1 : 0);
    sendcounts[proc] = proc_local_height * width_;
    if (proc == 0) {
      displacements[proc] = 0;
    } else {
      displacements[proc] = displacements[proc - 1] + sendcounts[proc - 1];
    }
  }

  MPI_Barrier(world);

  MPI_Gatherv(output_image_.data(), output_image_.size(), MPI_UNSIGNED_CHAR, gathered_output.data(), sendcounts.data(),
              displacements.data(), MPI_UNSIGNED_CHAR, 0, world);

  if (world.rank() == 0) {
    std::copy(gathered_output.begin(), gathered_output.end(), taskData->outputs[0]);
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

  std::copy(output_image_.begin(), output_image_.end(), taskData->outputs[0]);
  return true;
}