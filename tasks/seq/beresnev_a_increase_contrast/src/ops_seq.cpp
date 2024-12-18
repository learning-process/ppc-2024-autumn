// Copyright 2024 Nesterov Alexander
#include "seq/beresnev_a_increase_contrast/include/ops_seq.hpp"

#include <algorithm>
#include <string>
#include <vector>

bool beresnev_a_increase_contrast_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  inp_.assign(taskData->inputs[0] + pixel_data_start, taskData->inputs[0] + pixel_data_start + pixel_data_size);
  // std::cerr << inp_.size();
  factor = reinterpret_cast<double*>(taskData->inputs[1])[0];
  res_ = std::vector<uint8_t>(pixel_data_size);
  return true;
}

bool beresnev_a_increase_contrast_seq::TestTaskSequential::validation() {
  internal_order_test();
  auto* buffer = taskData->inputs[0];
  auto buffer_size = taskData->inputs_count[0];
  if (buffer == nullptr || buffer_size == 0) return false;

  std::string format(buffer, buffer + 2);
  size_t pos = 2;

  if (format != "P6") {
    std::cerr << "Error: Invalid file format (expected P6)." << std::endl;
    return false;
  }

  while (buffer[pos] == ' ' || buffer[pos] == '\n') ++pos;
  width = 0;
  while (buffer[pos] >= '0' && buffer[pos] <= '9') {
    width = width * 10 + (buffer[pos] - '0');
    ++pos;
  }

  while (buffer[pos] == ' ' || buffer[pos] == '\n') ++pos;
  height = 0;
  while (buffer[pos] >= '0' && buffer[pos] <= '9') {
    height = height * 10 + (buffer[pos] - '0');
    ++pos;
  }

  while (buffer[pos] == ' ' || buffer[pos] == '\n') ++pos;
  max_color = 0;
  while (buffer[pos] >= '0' && buffer[pos] <= '9') {
    max_color = max_color * 10 + (buffer[pos] - '0');
    ++pos;
  }

  if (max_color != 255) {
    std::cerr << "Error: Only maximum color value 255 is supported." << std::endl;
    return false;
  }
  ++pos;

  pixel_data_start = pos;
  pixel_data_size = width * height * 3;

  if (buffer_size != pixel_data_start + pixel_data_size) {
    std::cerr << "Error: The size of the pixel data does not match the total file size." << buffer_size << " "
              << pixel_data_start << " " << pixel_data_size << std::endl;
    return false;
  }
  return taskData->inputs_count[1] == 1 && taskData->outputs_count[0] == buffer_size;
}

bool beresnev_a_increase_contrast_seq::TestTaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < pixel_data_size; ++i) {
    double normalized = inp_[i] / 255.0;
    normalized = (normalized - 0.5) * factor + 0.5;
    normalized = std::clamp(normalized, 0.0, 1.0);
    res_[i] = static_cast<uint8_t>(normalized * 255);
  }
  return true;
}

bool beresnev_a_increase_contrast_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  std::vector<uint8_t> out_;
  out_.reserve(pixel_data_size + pixel_data_start);
  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  out_.insert(out_.end(), head.begin(), head.end());
  out_.insert(out_.end(), res_.data(), res_.data() + pixel_data_size);
  // std::cerr << out_.size();
  reinterpret_cast<std::vector<uint8_t>*>(taskData->outputs[0])[0] = out_;
  return true;
}