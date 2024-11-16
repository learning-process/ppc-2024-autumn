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
  unsigned int delta=0;
  if (world.rank() == 0) {
    auto input_size = taskData->inputs_count[0];
    auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_ = std::vector<int>(input_ptr, input_ptr + input_size);
    output_ = input_;
    auto total_brightness = 0;
    for (size_t i = 0; i < input_size; i += 3) {
      total_brightness += static_cast<int>(input_[i] * 0.299 + input_[i + 1] * 0.587 + input_[i + 2] * 0.114);
    }
    av_br = total_brightness / (input_size / 3);
    delta = input_size / world.size();
    broadcast(world, delta, 0);
    local_output_.resize(delta);
    if (world.rank() == 0) {
      for (int proc = 1; proc < world.size(); proc++) {
        world.send(proc, 0, input_.data() + proc * delta, delta);
      }
      local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
    }
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  return true;
}

bool kolodkin_g_image_contrast_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
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
  if (world.rank() == 0) {
    output_.resize(taskData->inputs_count[0]);
    std::copy(local_output_.begin(), local_output_.end(), output_.begin());
    for (int proc = 1; proc < world.size(); proc++) {
      int local_size = local_output_.size();
      std::vector<int> recv_buffer(local_size);
      world.recv(proc, 0, recv_buffer.data(), local_size);
      std::copy(recv_buffer.begin(), recv_buffer.end(), output_.begin() + proc * local_size);
    }
  } else {
    world.send(0, 0, local_output_.data(), local_output_.size());
  }
  return true;
}

bool kolodkin_g_image_contrast_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<std::vector<int>*>(taskData->outputs[0]) = output_;
  }
  return true;
}
