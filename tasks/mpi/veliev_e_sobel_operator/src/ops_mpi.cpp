// Copyright 2023 Nesterov Alexander
#include "mpi/veliev_e_sobel_operator/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

std::vector<double> sobel_filter(const std::vector<double>& image_vector, int h, int w) {
  std::vector<double> result(h * w, 0.0);

  const double sobel_x[3][3] = {{-1.0, 0.0, 1.0}, {-2.0, 0.0, 2.0}, {-1.0, 0.0, 1.0}};

  const double sobel_y[3][3] = {{-1.0, -2.0, -1.0}, {0.0, 0.0, 0.0}, {1.0, 2.0, 1.0}};

  for (int y = 1; y < h - 1; ++y) {
    for (int x = 1; x < w - 1; ++x) {
      double gx = 0.0;
      double gy = 0.0;

      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          int pixel_idx = (y + i) * w + (x + j);
          gx += image_vector[pixel_idx] * sobel_x[i + 1][j + 1];
          gy += image_vector[pixel_idx] * sobel_y[i + 1][j + 1];
        }
      }

      result[y * w + x] = std::sqrt(gx * gx + gy * gy);
    }
  }
  double max_val = *std::max_element(result.begin(), result.end());
  if (max_val > 0.0) {
    std::transform(result.begin(), result.end(), result.begin(), [max_val](double val) { return val / max_val; });
  }

  return result;
}

bool veliev_e_sobel_operator_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_.resize(taskData->inputs_count[0]);
  h = taskData->inputs_count[1];
  w = taskData->inputs_count[2];
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  return true;
}

bool veliev_e_sobel_operator_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[1] < 3 || taskData->inputs_count[2] < 3) return false;
  return taskData->inputs_count[0] > 0;
}

bool veliev_e_sobel_operator_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res_ = sobel_filter(input_, h, w);
  return true;
}

bool veliev_e_sobel_operator_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), tmp_ptr);
  return true;
}

bool veliev_e_sobel_operator_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_.resize(taskData->inputs_count[0]);
    h = taskData->inputs_count[1];
    w = taskData->inputs_count[2];
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
    res_.resize(taskData->inputs_count[0]);
  }
  return true;
}

bool veliev_e_sobel_operator_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[1] < 3 || taskData->inputs_count[2] < 3) return false;
  }
  return true;
}

bool veliev_e_sobel_operator_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  unsigned int d = 0;
  unsigned int r = 0;
  int width = 0;
  if (world.rank() == 0) {
    d = (h - 2) / world.size();
    r = (h - 2) % world.size();
    width = w;
  }
  broadcast(world, d, 0);
  broadcast(world, r, 0);
  broadcast(world, width, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + (proc * d + r) * width, (d + 2) * width);
    }
  }
  local_input_ = std::vector<double>((d + 2) * width);
  if (world.rank() == 0) {
    local_input_ = std::vector<double>(input_.begin(), input_.begin() + (d + r + 2) * width);
  } else {
    world.recv(0, 0, local_input_.data(), (d + 2) * width);
  }

  int height = static_cast<int>(local_input_.size()) / width;

  std::vector<double> res(height * width, 0.0);

  const double sobel_x[3][3] = {{-1.0, 0.0, 1.0}, {-2.0, 0.0, 2.0}, {-1.0, 0.0, 1.0}};
  const double sobel_y[3][3] = {{-1.0, -2.0, -1.0}, {0.0, 0.0, 0.0}, {1.0, 2.0, 1.0}};

  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      double gx = 0.0;
      double gy = 0.0;

      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          int pixel_idx = (y + i) * width + (x + j);
          gx += local_input_[pixel_idx] * sobel_x[i + 1][j + 1];
          gy += local_input_[pixel_idx] * sobel_y[i + 1][j + 1];
        }
      }

      res[y * width + x] = std::sqrt(gx * gx + gy * gy);
    }
  }
  res.erase(res.begin(), res.begin() + width);
  res.erase(res.end() - width, res.end());
  local_input_.clear();
  local_input_.assign(res.begin(), res.end());
  double local_max = *std::max_element(local_input_.begin(), local_input_.end());
  double global_max = -1.0;
  reduce(world, local_max, global_max, boost::mpi::maximum<double>(), 0);
  broadcast(world, global_max, 0);

  if (global_max > 0.0) {
    for (double& val : local_input_) {
      val /= global_max;
    }
  }
  std::vector<int> local_sizes(world.size(), d * width);
  local_sizes[0] += r * width;

  gatherv(world, local_input_.data(), local_input_.size(), res_.data(), local_sizes, 0);

  if (world.rank() == 0) {
    std::vector<double> zero_row(width, 0.0);
    res_.insert(res_.begin(), zero_row.begin(), zero_row.end());
    res_.erase(res_.end() - w, res_.end());
  }

  return true;
}

bool veliev_e_sobel_operator_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), tmp_ptr);
  }
  return true;
}
