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
    // Init vectors
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  }
  output_ = std::vector<int>(taskData->inputs_count[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    output_[i] = 0;
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
  unsigned int delta = taskData->inputs_count[0] / world.size();
  unsigned int last = taskData->inputs_count[0] % world.size();
  if (world.rank() == 0) {
    for (unsigned long i = 0; i < input_.size(); i = i + 3) {
      int ValueR = input_[i];
      int ValueG = input_[i + 1];
      int ValueB = input_[i + 2];
      av_br += (int)(ValueR * 0.299 + ValueG * 0.587 + ValueB * 0.114);
    }
    av_br /= input_.size() / 3;
    palette_.resize(256);
    double k = 1.5;
    for (int i = 0; i < 256; i++) {
      int delta_color = i - av_br;
      int temp = static_cast<int>(av_br + k * delta_color);
      palette_[i] = std::clamp(temp, 0, 255);
    }
  }
  broadcast(world, av_br, 0);
  std::vector<int> sizes;
  unsigned int localSize;
  for (unsigned int i = 0; i < (unsigned int)world.size(); i++) {
    localSize = delta;
    if (i < last) localSize++;
    sizes.push_back(localSize);
  }
  std::vector<int> local_input_(sizes[world.rank()]);
  std::vector<int> local_output_(sizes[world.rank()]);
  scatterv(world, input_, sizes, local_input_.data(), 0);
  for (size_t i = 0; i < local_output_.size(); i++) {
    local_output_[i] = palette_[local_input_[i]];
  }
  gatherv(world, local_output_.data(), local_output_.size(), output_.data(), sizes, 0);
  return true;
}

bool kolodkin_g_image_contrast_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<std::vector<int>*>(taskData->outputs[0]) = output_;
  }
  return true;
}
