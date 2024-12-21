#include "seq/sozonov_i_image_filtering_vertical_gaussian_3x3/include/ops_seq.hpp"

#include <numbers>
#include <thread>

using namespace std::chrono_literals;

bool sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init image
  image = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], image.begin());
  width = taskData->inputs_count[1];
  height = taskData->inputs_count[2];
  // Init filtered image
  filtered_image = std::vector<double>(width * height, 0);
  return true;
}

bool sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Init image
  image = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], image.begin());

  size_t img_size = taskData->inputs_count[1] * taskData->inputs_count[2];

  // Check pixels range from 0 to 255
  for (size_t i = 0; i < img_size; ++i) {
    if (image[i] < 0 || image[i] > 255) {
      return false;
    }
  }

  // Check size of image
  return taskData->inputs_count[0] == img_size && taskData->outputs_count[0] == img_size &&
         taskData->inputs_count[1] >= 3 && taskData->inputs_count[2] >= 3;
}

bool sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential::run() {
  internal_order_test();
  std::vector<double> kernel = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
  std::vector<double> tmp_image((width + 2) * height, 0);
  for (int i = 0; i < height; ++i) {
    for (int j = 1; j < width + 1; ++j) {
      tmp_image[i * (width + 2) + j] = image[i * width + j - 1];
    }
  }
  std::vector<double> tmp_filtered_image((width + 2) * height, 0);
  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width + 1; ++j) {
      double sum = 0;
      for (int l = -1; l <= 1; ++l) {
        for (int k = -1; k <= 1; ++k) {
          sum += tmp_image[(i - l) * (width + 2) + j - k] * kernel[(l + 1) * 3 + k + 1];
        }
      }
      tmp_filtered_image[i * (width + 2) + j] = sum;
    }
  }
  for (int i = 0; i < height; ++i) {
    for (int j = 1; j < width + 1; ++j) {
      filtered_image[i * width + j - 1] = tmp_filtered_image[i * (width + 2) + j];
    }
  }
  return true;
}

bool sozonov_i_image_filtering_vertical_gaussian_3x3_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* out = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(filtered_image.begin(), filtered_image.end(), out);
  return true;
}