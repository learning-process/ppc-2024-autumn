#include "mpi/fomin_v_sobel_edges/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
using namespace std::chrono_literals;

#include <boost/mpi.hpp>

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_image_ = *reinterpret_cast<std::vector<unsigned char>*>(taskData->inputs[0]);
    width_ = taskData->inputs_count[0];
    height_ = taskData->inputs_count[1];
  }

  int local_height;
  if (world.rank() == 0) {
    local_height = height_ / world.size();
    if (height_ % world.size() != 0) {
      local_height += 1;
    }
  }
  MPI_Bcast(&width_, 1, MPI_INT, 0, world);
  MPI_Bcast(&height_, 1, MPI_INT, 0, world);

  MPI_Bcast(&local_height, 1, MPI_INT, 0, world);

  local_input_.resize((local_height + 2) * width_, 0);  // +2 for boundary rows
  output_image_.resize(local_height * width_, 0);
  if (world.rank() == 0) {
    std::cout << "Rank 0: width = " << width_ << ", height = " << height_ << std::endl;
  }

  std::cout << "Rank " << world.rank() << ": local_height = " << local_height << std::endl;
  if (world.rank() == 0) {
    std::cout << "Rank 0: local_input_ size = " << local_input_.size() << std::endl;
    for (int i = 0; i < 10; ++i) {
      std::cout << "Rank 0: local_input_[" << i << "] = " << static_cast<int>(local_input_[i]) << std::endl;
    }
    int disp = 0;
    for (int proc = 0; proc < world.size(); ++proc) {
      int proc_local_height = local_height;
      if (proc == world.size() - 1) {
        proc_local_height = height_ - (world.size() - 1) * local_height;
      }
      if (proc != 0) {
        world.send(proc, 0, input_image_.data() + disp - width_, width_ * (proc_local_height + 2));
      } else {
        world.send(proc, 0, input_image_.data() + disp, width_ * (proc_local_height + 2));
      }
      disp += proc_local_height * width_;
    }
  } else {
    world.recv(0, 0, local_input_.data(), width_ * (local_height + 2));
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
  for (int y = 1; y <= local_height; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      int sumX = 0;
      int sumY = 0;
      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          int pixel = local_input_[(y + i) * width_ + (x + j)];
          sumX += pixel * Gx[i + 1][j + 1];
          sumY += pixel * Gy[i + 1][j + 1];
        }
      }
      int gradient = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
      output_image_[(y - 1) * width_ + x] = static_cast<unsigned char>(std::min(gradient, 255));
      if (y == 1 && x == 1) {  // Print for a few specific pixels
        std::cout << "Rank " << world.rank() << ": Calculated gradient at (y,x) = (" << y << "," << x
                  << ") = " << static_cast<int>(output_image_[(y - 1) * width_ + x]) << std::endl;
      }
    }
  }
  return true;
}

bool fomin_v_sobel_edges::SobelEdgeDetectionMPI::post_processing() {
  internal_order_test();

  internal_order_test();

  std::vector<int> sendcounts(world.size());
  std::vector<int> displacements(world.size());
  std::vector<unsigned char> gathered_output;

  if (world.rank() == 0) {
    for (int proc = 0; proc < world.size(); ++proc) {
      int proc_local_height = (height_ + world.size() - 1 - proc) / world.size();
      sendcounts[proc] = proc_local_height * width_;
      if (proc == 0) {
        displacements[proc] = 0;
      } else {
        displacements[proc] = displacements[proc - 1] + sendcounts[proc - 1];
      }
    }
    gathered_output.resize(height_ * width_, 0);
  }

  MPI_Gatherv(output_image_.data(), output_image_.size(), MPI_UNSIGNED_CHAR, gathered_output.data(), sendcounts.data(),
              displacements.data(), MPI_UNSIGNED_CHAR, 0, world);

  if (world.rank() == 0) {
    *reinterpret_cast<std::vector<unsigned char>*>(taskData->outputs[0]) = gathered_output;
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