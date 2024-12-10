// Copyright 2024 Anikin Maksim
#include "mpi/anikin_m_contrastscale/include/ops_mpi.hpp"

#include <algorithm>
#include <random>

anikin_m_contrastscale_mpi::RGB anikin_m_contrastscale_mpi::getrandomRGB() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<int> dist(0, 255);
  RGB rgb;
  rgb.R = dist(gen);
  rgb.G = dist(gen);
  rgb.B = dist(gen);
  return rgb;
}

double anikin_m_contrastscale_mpi::getcontrast(std::vector<anikin_m_contrastscale_mpi::RGB> &in) {
  auto [min, max] = std::minmax_element(in.begin(), in.end(), [](auto a, auto b) { return a.R < b.R; });
  return (double)(max->R - min->R) / (max->R + min->R);
}

bool anikin_m_contrastscale_mpi::ContrastScaleSeq::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] >= 1) && (taskData->inputs_count[0] == taskData->outputs_count[0]);
}

bool anikin_m_contrastscale_mpi::ContrastScaleSeq::pre_processing() {
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

bool anikin_m_contrastscale_mpi::ContrastScaleSeq::run() {
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

bool anikin_m_contrastscale_mpi::ContrastScaleSeq::post_processing() {
  internal_order_test();
  auto *output_ptr = reinterpret_cast<RGB *>(taskData->outputs[0]);
  std::copy(output_.begin(), output_.end(), output_ptr);
  return true;
}

bool anikin_m_contrastscale_mpi::ContrastScaleMpi::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs_count[0] >= 1) && (taskData->inputs_count[0] == taskData->outputs_count[0]);
  }
  return true;
}

bool anikin_m_contrastscale_mpi::ContrastScaleMpi::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto *input_ptr = reinterpret_cast<RGB *>(taskData->inputs[0]);
    correction = *reinterpret_cast<float *>(taskData->inputs[1]);
    input_.assign(input_ptr, input_ptr + taskData->inputs_count[0]);
    output_.resize(taskData->inputs_count[0]);
  }
  return true;
}

bool anikin_m_contrastscale_mpi::ContrastScaleMpi::run() {
  internal_order_test();
  // Init input_size
  uint32_t input_size;
  if (world.rank() == 0) {
    input_size = taskData->inputs_count[0];
    output_.clear();
    output_.resize(taskData->inputs_count[0]);
  }
  // BC data
  broadcast(world, input_size, 0);
  broadcast(world, correction, 0);
  // Init local sizes
  int size_dec = input_size / world.size();
  int size_mod = input_size % world.size();
  std::vector<int> local_sizes;
  for (int i = 0; i < world.size(); i++) {
    int local_size = size_dec;
    if (i < size_mod) {
      local_size++;
    }
    local_sizes.push_back(local_size);
  }
  // Init local input
  std::vector<anikin_m_contrastscale_mpi::RGB> local_input(local_sizes[world.rank()]);
  scatterv(world, input_, local_sizes, local_input.data(), 0);
  // Calculate iab
  int iab = 0;
  for (auto i : local_input) {
    iab += (int)(i.R * 0.299 + i.G * 0.587 + i.B * 0.114);
  }
  iab = all_reduce(world, iab, std::plus());
  iab = iab / input_size;
  // Calculate newRGB
  uint8_t newrgb[256] = {0};
  for (int i = 0; i < 256; i++) {
    int delta = i - iab;
    int temp = (int)(iab + correction * delta);

    if (temp < 0) temp = 0;
    if (temp >= 255) temp = 255;

    newrgb[i] = (uint8_t)temp;
  }
  // Create new local images
  std::vector<anikin_m_contrastscale_mpi::RGB> local_output;
  for (auto i : local_input) {
    RGB newp;
    newp.R = newrgb[i.R];
    newp.G = newrgb[i.G];
    newp.B = newrgb[i.B];
    local_output.push_back(newp);
  }
  // Collect all data
  gatherv(world, local_output.data(), local_output.size(), output_.data(), local_sizes, 0);
  return true;
}

bool anikin_m_contrastscale_mpi::ContrastScaleMpi::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto *output_ptr = reinterpret_cast<RGB *>(taskData->outputs[0]);
    std::copy(output_.begin(), output_.end(), output_ptr);
  }
  return true;
}