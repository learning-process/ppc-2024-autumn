#include "mpi/kolodkin_g_image_contrast/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool kolodkin_g_image_contrast_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  auto input_size = taskData->inputs_count[0];
  input_ = std::vector<int>(input_ptr, input_ptr + input_size);
  output_ = input_;
  palette_.resize(256);
  return true;
}

bool kolodkin_g_image_contrast_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] <= 0 || taskData->inputs_count[0] % 3 != 0) {
    return false;
  }
  auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned long i = 0; i < taskData->inputs_count[0]; i++) {
    if (*input_ptr > 255 || *input_ptr < 0) {
      return false;
    }
    input_ptr++;
  }
  return true;
}

bool kolodkin_g_image_contrast_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (unsigned long i = 0; i < input_.size(); i = i + 3) {
    int ValueR = input_[i];
    int ValueG = input_[i + 1];
    int ValueB = input_[i + 2];
    av_br += (int)(ValueR * 0.299 + ValueG * 0.587 + ValueB * 0.114);
  }
  av_br /= input_.size() / 3;
  double k = 1.5;
  for (int i = 0; i < 256; i++) {
    int delta_color = (int)i - av_br;
    int temp = (int)(av_br + k * delta_color);
    palette_[i] = std::clamp(temp, 0, 255);
  }
  for (unsigned long i = 0; i < input_.size(); i++) {
    int value = input_[i];
    output_[i] = palette_[value];
  }
  return true;
}

bool kolodkin_g_image_contrast_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<std::vector<int>*>(taskData->outputs[0]) = output_;
  return true;
}

bool kolodkin_g_image_contrast_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto input_size = taskData->inputs_count[0];
    auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_ = std::vector<int>(input_ptr, input_ptr + input_size);
    output_ = input_;
    local_output_.resize(input_.size());
    auto total_brightness = 0;
    for (size_t i = 0; i < input_size; i += 3) {
      total_brightness += static_cast<int>(input_[i] * 0.299 + input_[i + 1] * 0.587 + input_[i + 2] * 0.114);
    }
    av_br = total_brightness / (input_size / 3);
    auto delta = input_size / world.size();
    for (int proc = 1; proc < world.size(); proc++) {
      auto start = proc * delta;
      auto size = (proc == world.size() - 1) ? input_size - start : delta;
      world.send(proc, 0, std::vector<int>(input_ptr + start, input_ptr + start + size));
    }
    local_input_ = std::vector<int>(input_ptr, input_ptr + delta);
  } else {
    int input_data_size;
    world.recv(0, 0, input_data_size);
    local_input_.resize(input_data_size);
    world.recv(0, 1, local_input_);
  }
  return true;
}

bool kolodkin_g_image_contrast_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] <= 0 || taskData->inputs_count[0] % 3 != 0) {
    return false;
  }
  auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned long i = 0; i < taskData->inputs_count[0]; i++) {
    if (*input_ptr > 255 || *input_ptr < 0) {
      return false;
    }
    input_ptr++;
  }
  return true;
}

bool kolodkin_g_image_contrast_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  std::vector<int> palette(256);
  double k = 1.5;
  for (int i = 0; i < 256; i++) {
    int delta_color = i - av_br;
    int temp = static_cast<int>(av_br + k * delta_color);
    palette[i] = std::clamp(temp, 0, 255);
  }
  local_output_.resize(local_input_.size());
  for (size_t i = 0; i < local_input_.size(); i++) {
    local_output_[i] = palette[local_input_[i]];
  }
  return true;
}

bool kolodkin_g_image_contrast_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < local_output_.size(); i++) {
      output_[i] = local_output_[i];
    }
    *reinterpret_cast<std::vector<int>*>(taskData->outputs[0]) = output_;
  }
  return true;
}
