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

  uint8_t local_min_val = *std::min_element(local_input.begin(), local_input.end());
  uint8_t local_max_val = *std::max_element(local_input.begin(), local_input.end());

  uint8_t global_min_val, global_max_val;
  boost::mpi::reduce(world, local_min_val, global_min_val, boost::mpi::minimum<uint8_t>(), 0);
  boost::mpi::reduce(world, local_max_val, global_max_val, boost::mpi::maximum<uint8_t>(), 0);

  if (world.rank() == 0) {
    if (global_min_val == global_max_val) {
      std::fill(output_.begin(), output_.end(), 255);
    } else {
      for (size_t i = 0; i < input_.size(); ++i) {
        output_[i] = static_cast<uint8_t>((input_[i] - global_min_val) * 255.0 / (global_max_val - global_min_val));
      }
    }
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