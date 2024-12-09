// Copyright 2023 Nesterov Alexander
#include "mpi/kondratev_ya_contrast_adjustment/include/ops_mpi.hpp"

double kondratev_ya_contrast_adjustment_mpi::getContrast(
    std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel>& array) {
  auto [min, max] = std::minmax_element(array.begin(), array.end(), [](auto a, auto b) { return a.red < b.red; });

  return (double)(max->red - min->red) / (max->red + min->red);
}

bool kondratev_ya_contrast_adjustment_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  auto* ptr = reinterpret_cast<Pixel*>(taskData->inputs[0]);
  contrast_ = *reinterpret_cast<double*>(taskData->inputs[1]);

  input_.assign(ptr, ptr + taskData->inputs_count[0]);
  res_.resize(taskData->inputs_count[0]);

  return true;
}

bool kondratev_ya_contrast_adjustment_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == taskData->inputs_count[0] && !taskData->outputs.empty() &&
         taskData->inputs.size() == 2 && *reinterpret_cast<double*>(taskData->inputs[1]) >= 0;
}

bool kondratev_ya_contrast_adjustment_mpi::TestMPITaskSequential::run() {
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

bool kondratev_ya_contrast_adjustment_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  auto* ptr = reinterpret_cast<Pixel*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), ptr);

  return true;
}

bool kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* ptr = reinterpret_cast<Pixel*>(taskData->inputs[0]);
    contrast_ = *reinterpret_cast<double*>(taskData->inputs[1]);

    input_.assign(ptr, ptr + taskData->inputs_count[0]);
    res_.resize(taskData->inputs_count[0]);
  }

  return true;
}

bool kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0)
    return taskData->outputs_count[0] == taskData->inputs_count[0] && !taskData->outputs.empty() &&
           taskData->inputs.size() == 2 && *reinterpret_cast<double*>(taskData->inputs[1]) >= 0;

  return true;
}

bool kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel::run() {
  internal_order_test();

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

  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> local_input(sizes[world.rank()]);
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> local_res(sizes[world.rank()]);
  scatterv(world, input_, sizes, local_input.data(), 0);

  std::vector<int> local_sum(3, 0);
  std::vector<int> sum(3);

  for (uint32_t i = 0; i < local_input.size(); i++) {
    local_sum[0] += local_input[i].red;
    local_sum[1] += local_input[i].green;
    local_sum[2] += local_input[i].blue;
  }

  all_reduce(world, local_sum.data(), local_sum.size(), sum.data(), std::plus());

  std::vector<double> average(3);
  auto size = static_cast<double>(step * world.size() + remain);
  for (uint32_t i = 0; i < 3; i++) average[i] = sum[i] / size;

  for (uint32_t i = 0; i < local_input.size(); i++) {
    local_res[i].red = std::clamp((int32_t)(contrast_ * (local_input[i].red - average[0]) + average[0]), 0, 255);
    local_res[i].green = std::clamp((int32_t)(contrast_ * (local_input[i].green - average[1]) + average[1]), 0, 255);
    local_res[i].blue = std::clamp((int32_t)(contrast_ * (local_input[i].blue - average[2]) + average[2]), 0, 255);
  }

  gatherv(world, local_res.data(), local_res.size(), res_.data(), sizes, 0);

  return true;
}

bool kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* ptr = reinterpret_cast<Pixel*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), ptr);
  }

  return true;
}
