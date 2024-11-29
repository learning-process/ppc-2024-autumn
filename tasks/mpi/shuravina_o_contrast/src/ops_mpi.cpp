#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <cmath>
#include <iostream>

bool shuravina_o_contrast::ContrastParallel::pre_processing() {
  if (world.rank() == 0) {
    auto* ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    contrast_ = *reinterpret_cast<double*>(taskData->inputs[1]);

    input_.assign(ptr, ptr + taskData->inputs_count[0]);
    output_.resize(taskData->inputs_count[0]);
  }

  return true;
}

bool shuravina_o_contrast::ContrastParallel::validation() {
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == taskData->inputs_count[0] && !taskData->outputs.empty() &&
           taskData->inputs.size() == 2 && *reinterpret_cast<double*>(taskData->inputs[1]) >= 0;
  }
  return true;
}

bool shuravina_o_contrast::ContrastParallel::run() {
  uint32_t inputSize;

  if (world.rank() == 0) {
    inputSize = input_.size();
  }

  broadcast(world, inputSize, 0);
  broadcast(world, contrast_, 0);

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

  std::vector<int> local_sum(1, 0);
  std::vector<int> sum(1);

  for (uint32_t i = 0; i < local_input.size(); i++) {
    local_sum[0] += local_input[i];
  }

  all_reduce(world, local_sum.data(), local_sum.size(), sum.data(), std::plus());

  double average = sum[0] / static_cast<double>(inputSize);

  for (uint32_t i = 0; i < local_input.size(); i++) {
    local_output[i] = std::clamp((int32_t)(contrast_ * (local_input[i] - average) + average), 0, 255);
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