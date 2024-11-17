#include "seq/gnitienko_k_contrast_enhancement/include/ops_seq.hpp"

#include <random>

std::vector<int> gnitienko_k_contrast_enhancement_seq::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 255;
  }
  return vec;
}

bool gnitienko_k_contrast_enhancement_seq::ContrastEnhanceSeq::pre_processing() {
  internal_order_test();

  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  contrast_factor = *reinterpret_cast<double*>(taskData->inputs[1]);
  size_t input_size = taskData->inputs_count[0];
  image.resize(input_size, 0);
  res.resize(input_size, 0);
  image.assign(input_data, input_data + input_size);

  return true;
}

bool gnitienko_k_contrast_enhancement_seq::ContrastEnhanceSeq::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == taskData->outputs_count[0] && taskData->inputs_count[0] >= 0;
}

bool gnitienko_k_contrast_enhancement_seq::ContrastEnhanceSeq::is_grayscale() const { return image.size() % 3 != 0; }

bool gnitienko_k_contrast_enhancement_seq::ContrastEnhanceSeq::run() {
  internal_order_test();
  if (is_grayscale()) {
    for (size_t i = 0; i < image.size(); ++i) {
      res[i] = std::clamp(static_cast<int>((image[i] - 128) * contrast_factor + 128), 0, 255);
    }
  } else {
    for (size_t i = 0; i < image.size(); i += 3) {
      res[i] = std::clamp(static_cast<int>((image[i] - 128) * contrast_factor + 128), 0, 255);
      res[i + 1] = std::clamp(static_cast<int>((image[i + 1] - 128) * contrast_factor + 128), 0, 255);
      res[i + 2] = std::clamp(static_cast<int>((image[i + 2] - 128) * contrast_factor + 128), 0, 255);
    }
  }
  return true;
}

bool gnitienko_k_contrast_enhancement_seq::ContrastEnhanceSeq::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < res.size(); ++i) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}
