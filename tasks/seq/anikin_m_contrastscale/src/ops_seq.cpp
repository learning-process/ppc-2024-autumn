// Copyright 2024 Anikin Maksim
#include "seq/anikin_m_contrastscale/include/ops_seq.hpp"

#include <thread>

anikin_m_contrastscale_seq::RGB anikin_m_contrastscale_seq::getrandomRGB() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<int> dist(0, 255);
  RGB rgb;
  rgb.R = dist(gen);
  rgb.G = dist(gen);
  rgb.B = dist(gen);
  return rgb;
}

double anikin_m_contrastscale_seq::getcontrast(std::vector<anikin_m_contrastscale_seq::RGB> &in) {
  auto [min, max] = std::minmax_element(in.begin(), in.end(), [](auto a, auto b) { return a.R < b.R; });
  return (double)(max->R - min->R) / (max->R + min->R);
}

bool anikin_m_contrastscale_seq::ContrastScaleSeq::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] >= 1) && (taskData->inputs_count[0] == taskData->outputs_count[0]);
}

bool anikin_m_contrastscale_seq::ContrastScaleSeq::pre_processing() {
  internal_order_test();
  auto *input_ptr = reinterpret_cast<RGB *>(taskData->inputs[0]);
  correction = *reinterpret_cast<float *>(taskData->inputs[1]);
  input_.assign(input_ptr, input_ptr + taskData->inputs_count[0]);
  output_.clear();
  iab = 0;
  for (auto i : input_) {
    iab += (int)(i.R * 0.299 + i.G * 0.587 + i.B * 0.114);
  }
  iab /= taskData->inputs_count[0];
  return true;
}

bool anikin_m_contrastscale_seq::ContrastScaleSeq::run() {
  internal_order_test();
  output_.clear();
  // Calculate new RGB
  uint8_t newrgb[256] = {0};
  for (int i = 0; i < 256; i++) {
    int delta = i - iab;
    int temp = (int)(iab + correction * delta);

    if (temp < 0) temp = 0;
    if (temp >= 255) temp = 255;

    newrgb[i] = (uint8_t)temp;
  }
  // Create new Image
  for (auto i : input_) {
    RGB newp;
    newp.R = newrgb[i.R];
    newp.G = newrgb[i.G];
    newp.B = newrgb[i.B];
    output_.push_back(newp);
  }
  return true;
}

bool anikin_m_contrastscale_seq::ContrastScaleSeq::post_processing() {
  internal_order_test();
  auto *output_ptr = reinterpret_cast<RGB *>(taskData->outputs[0]);
  std::copy(output_.begin(), output_.end(), output_ptr);
  return true;
}