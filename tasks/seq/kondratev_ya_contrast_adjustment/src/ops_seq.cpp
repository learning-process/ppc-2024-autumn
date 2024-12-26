// Copyright 2024 Nesterov Alexander
#include "seq/kondratev_ya_contrast_adjustment/include/ops_seq.hpp"

double kondratev_ya_contrast_adjustment_seq::getContrast(
    std::vector<kondratev_ya_contrast_adjustment_seq::Pixel>& array) {
  auto [min, max] = std::minmax_element(array.begin(), array.end(), [](auto a, auto b) { return a.red < b.red; });

  return (double)(max->red - min->red) / (max->red + min->red);
}

bool kondratev_ya_contrast_adjustment_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  auto* ptr = reinterpret_cast<Pixel*>(taskData->inputs[0]);
  contrast_ = *reinterpret_cast<double*>(taskData->inputs[1]);

  input_.assign(ptr, ptr + taskData->inputs_count[0]);
  res_.resize(taskData->inputs_count[0]);

  return true;
}

bool kondratev_ya_contrast_adjustment_seq::TestTaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == taskData->inputs_count[0] && !taskData->outputs.empty() &&
         taskData->inputs.size() == 2 && *reinterpret_cast<double*>(taskData->inputs[1]) >= 0;
}

bool kondratev_ya_contrast_adjustment_seq::TestTaskSequential::run() {
  internal_order_test();

  double average[3]{0, 0, 0};  // RGB

  for (auto& pixel : input_) {
    average[0] += pixel.red;
    average[1] += pixel.green;
    average[2] += pixel.blue;
  }

  for (uint32_t i = 0; i < 3; i++) average[i] /= input_.size();

  for (uint32_t i = 0; i < input_.size(); i++) {
    res_[i].red = std::clamp((int32_t)(contrast_ * (input_[i].red - average[0]) + average[0]), 0, 255);
    res_[i].green = std::clamp((int32_t)(contrast_ * (input_[i].green - average[1]) + average[1]), 0, 255);
    res_[i].blue = std::clamp((int32_t)(contrast_ * (input_[i].blue - average[2]) + average[2]), 0, 255);
  }

  return true;
}

bool kondratev_ya_contrast_adjustment_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  auto* ptr = reinterpret_cast<Pixel*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), ptr);

  return true;
}
