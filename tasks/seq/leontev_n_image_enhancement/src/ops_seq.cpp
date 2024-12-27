#include "seq/leontev_n_image_enhancement/include/ops_seq.hpp"

#include <random>
#include <thread>

bool leontev_n_image_enhancement_seq::ImgEnhancementSequential::pre_processing() {
  internal_order_test();

  int size = taskData->inputs_count[0];
  image_input = std::vector<int>(size);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + size, image_input.begin());

  int pixel_count = size / 3;
  I.resize(pixel_count);
  for (int i = 0; i < pixel_count; i++) {
    int r = image_input[3 * i];
    int g = image_input[3 * i + 1];
    int b = image_input[3 * i + 2];

    I[i] = (r + g + b) / 3;
  }

  image_output = {};
  return true;
}

bool leontev_n_image_enhancement_seq::ImgEnhancementSequential::validation() {
  internal_order_test();
  int size = taskData->inputs_count[0];
  if (size % 3 != 0) return false;

  for (int i = 0; i < size; ++i) {
    int value = reinterpret_cast<int*>(taskData->inputs[0])[i];
    if (value < 0 || value > 255) {
      return false;
    }
  }

  return (taskData->inputs_count[0] > 0) && (taskData->outputs_count[0] > 0);
}

bool leontev_n_image_enhancement_seq::ImgEnhancementSequential::run() {
  internal_order_test();
  int size = image_input.size();
  image_output.resize(size);
  int Imin = 255;
  int Imax = 0;
  for (int intensity : I) {
    Imin = std::min(Imin, intensity);
    Imax = std::max(Imax, intensity);
  }
  if (Imin == Imax) {
    image_output = image_input;
    return true;
  }
  for (size_t i = 0; i < I.size(); i++) {
    int Inew = ((I[i] - Imin) * 255) / (Imax - Imin);
    float scale = static_cast<float>(Inew) / static_cast<float>(I[i]);
    image_output[3 * i] = std::min(255, static_cast<int>(image_input[3 * i] * scale));
    image_output[3 * i + 1] = std::min(255, static_cast<int>(image_input[3 * i + 1] * scale));
    image_output[3 * i + 2] = std::min(255, static_cast<int>(image_input[3 * i + 2] * scale));
  }
  return true;
}

bool leontev_n_image_enhancement_seq::ImgEnhancementSequential::post_processing() {
  internal_order_test();
  auto* output = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(image_output.begin(), image_output.end(), output);
  return true;
}