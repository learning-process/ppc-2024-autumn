// Copyright 2024 Nesterov Alexander
#include "seq/veliev_e_sobel_operator/include/ops_seq.hpp"

std::vector<double> sobel_filter(const std::vector<double> &image_vector, int h, int w) {
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
    for (double &val : result) {
      val /= max_val;
    }
  }

  return result;
}

bool veliev_e_sobel_operator_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_.resize(taskData->inputs_count[0]);
  h = taskData->inputs_count[1];
  w = taskData->inputs_count[2];
  auto *tmp_ptr = reinterpret_cast<double *>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  return true;
}

bool veliev_e_sobel_operator_seq::TestTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[1] < 3 || taskData->inputs_count[2] < 3) return false;
  return taskData->inputs_count[0] > 0;
}

bool veliev_e_sobel_operator_seq::TestTaskSequential::run() {
  internal_order_test();
  res_ = sobel_filter(input_, h, w);
  return true;
}

bool veliev_e_sobel_operator_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto *tmp_ptr = reinterpret_cast<double *>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), tmp_ptr);
  return true;
}
