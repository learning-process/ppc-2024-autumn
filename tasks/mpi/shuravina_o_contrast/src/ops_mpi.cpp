#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <cmath>
#include <iostream>

bool shuravina_o_contrast::ContrastParallel::pre_processing() {
  if (world.rank() == 0) {
    auto* ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    input_.assign(ptr, ptr + taskData->inputs_count[0]);
    output_.resize(taskData->inputs_count[0]);
  }

  return true;
}

bool shuravina_o_contrast::ContrastParallel::validation() {
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == taskData->inputs_count[0] && !taskData->outputs.empty();
  }
  return true;
}

bool shuravina_o_contrast::ContrastParallel::run() {
  uint32_t inputSize;

  if (world.rank() == 0) {
    inputSize = input_.size();
  }

  broadcast(world, inputSize, 0);

  uint32_t step = inputSize / world.size();
  uint32_t remain = inputSize % world.size();

  std::vector<int> sizes;
  uint32_t recvSize;
  for (uint32_t i = 0; i < (uint32_t)world.size(); i++) {
    recvSize = step;
    if (i < remain) recvSize++;
    sizes.push_back(recvSize);
  }

  std::vector<uint8_t> local_input(sizes[world.rank()]);
  std::vector<uint8_t> local_output(sizes[world.rank()]);
  scatterv(world, input_, sizes, local_input.data(), 0);

  double gamma = 0.5;

  for (size_t i = 0; i < local_input.size(); ++i) {
    double normalized_value = static_cast<double>(local_input[i]) / 255.0;
    double corrected_value = std::pow(normalized_value, gamma);
    local_output[i] = static_cast<uint8_t>(corrected_value * 255.0);
  }

  gatherv(world, local_output.data(), local_output.size(), output_.data(), sizes, 0);

  return true;
}

bool shuravina_o_contrast::ContrastParallel::post_processing() {
  if (world.rank() == 0) {
    auto* ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
    std::copy(output_.begin(), output_.end(), ptr);
  }

  return true;
}