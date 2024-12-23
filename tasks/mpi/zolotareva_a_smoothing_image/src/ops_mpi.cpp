// Copyright 2023 Nesterov Alexander
// здесь писать саму задачу
#include "mpi/zolotareva_a_smoothing_image/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <seq/zolotareva_a_smoothing_image/include/ops_seq.hpp>
#include <vector>
std::vector<float> zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::create_gaussian_kernel(int radius,
                                                                                                   float sigma) {
  int size = 2 * radius + 1;
  std::vector<float> kernel(size);
  float norm = 0.0f;
  for (int i = -radius; i <= radius; ++i) {
    kernel[i + radius] = std::exp(-(i * i) / (2 * sigma * sigma));
    norm += kernel[i + radius];
  }
  for (float& val : kernel) {
    val /= norm;
  }
  return kernel;
}

void zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::convolve_rows(const std::vector<uint8_t>& input,
                                                                            int height, int width,
                                                                            const std::vector<float>& kernel,
                                                                            std::vector<float>& temp) {
  int kernel_radius = kernel.size() / 2;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -kernel_radius; k <= kernel_radius; ++k) {
        int pixel_x = std::clamp(x + k, 0, width - 1);
        sum += input[y * width + pixel_x] * kernel[k + kernel_radius];
      }
      temp[y * width + x] = sum;
    }
  }
}

void zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::convolve_columns(const std::vector<float>& temp,
                                                                               int height, int width,
                                                                               const std::vector<float>& kernel,
                                                                               std::vector<uint8_t>& output) {
  int kernel_radius = kernel.size() / 2;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -kernel_radius; k <= kernel_radius; ++k) {
        int pixel_y = std::clamp(y + k, 0, height - 1);
        sum += temp[pixel_y * width + x] * kernel[k + kernel_radius];
      }
      output[y * width + x] = static_cast<uint8_t>(std::clamp(sum, 0.0f, 255.0f));
      ;
    }
  }
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count.size() == 2 && taskData->inputs_count[0] > 1 && taskData->inputs_count[1] > 0;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  height_ = taskData->inputs_count[0];
  width_ = taskData->inputs_count[1];
  input_.resize(height_ * width_);
  const uint8_t* raw_data = reinterpret_cast<uint8_t*>(taskData->inputs[0]);

  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      input_[i * width_ + j] = raw_data[i * width_ + j];
    }
  }
  result_.resize(height_ * width_);
  return true;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  int radius = 1;
  float sigma = 1.5f;
  std::vector<float> horizontal_kernel = create_gaussian_kernel(radius, sigma);
  const std::vector<float>& vertical_kernel = horizontal_kernel;
  std::vector<float> temp(height_ * width_, 0.0f);
  convolve_rows(input_, height_, width_, horizontal_kernel, temp);
  convolve_columns(temp, height_, width_, vertical_kernel, result_);

  return true;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* output_raw = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
  for (int i = 0; i < height_; i++) {
    for (int j = 0; j < width_; j++) {
      output_raw[i * width_ + j] = result_[i * width_ + j];
    }
  }
  return true;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count.size() == 2 && taskData->inputs_count[0] > 1 &&
           taskData->inputs_count[0] >= size_t(world.size()) && taskData->inputs_count[1] > 0;
  }
  return true;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    height_ = taskData->inputs_count[0];
    width_ = taskData->inputs_count[1];
    size_ = height_ * width_;
    input_.resize(size_);
    const uint8_t* raw_data = taskData->inputs[0];
    std::copy(raw_data, raw_data + size_, input_.begin());
  }

  return true;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int world_size = world.size();
  int base_height = height_ / world_size;
  if (world_size > 1) {
    std::vector<int> displs(world_size);
    boost::mpi::broadcast(world, size_, 0);
    boost::mpi::broadcast(world, width_, 0);

    if (world.rank() > 0) send_counts.resize(world_size);
    {
      input_.resize(size_);
      displs.resize(world_size);
    }

    if (world.rank() == 0) {
      int remainder = height_ % world_size;
      int send_start = (base_height + remainder - 1) * width_;
      send_counts.resize(world_size, (base_height + 2) * width_);
      send_counts[0] = (base_height + remainder + 1) * width_;
      send_counts[world_size - 1] = (base_height + 1) * width_;

      displs[0] = 0;
      for (int proc = 1; proc < (world_size - 1); ++proc) {
        displs[proc] = send_start;
        send_start += base_height * width_;
      }
      displs[world_size - 1] = send_start;
    }

    boost::mpi::broadcast(world, send_counts.data(), send_counts.size(), 0);
    boost::mpi::broadcast(world, displs.data(), displs.size(), 0);
    boost::mpi::broadcast(world, input_.data(), input_.size(), 0);
    local_input_.resize(send_counts[world.rank()]);
    local_height_ = send_counts[world.rank()] / width_;
    boost::mpi::scatterv(world, input_.data(), send_counts, displs, local_input_.data(), send_counts[world.rank()], 0);
  }
  std::vector<uint8_t> local_res(local_height_ * width_);

  std::vector<float> horizontal_kernel(3);
  if (world.rank() == 0) {
    horizontal_kernel = zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::create_gaussian_kernel(1, 1.5f);
  }
  if (world_size == 1) {
    result_.resize(height_ * width_);
    std::vector<float>& vertical_kernel = horizontal_kernel;
    std::vector<float> temp(height_ * width_, 0.0f);
    zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::convolve_rows(input_, height_, width_, horizontal_kernel,
                                                                           temp);

    zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::convolve_columns(temp, height_, width_, vertical_kernel,
                                                                              result_);
    return true;
  }
  boost::mpi::broadcast(world, horizontal_kernel.data(), 3, 0);
  std::vector<float>& vertical_kernel = horizontal_kernel;
  std::vector<float> temp(local_height_ * width_, 0.0f);
  zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::convolve_rows(local_input_, local_height_, width_,
                                                                         horizontal_kernel, temp);

  zolotareva_a_smoothing_image_mpi::TestMPITaskSequential::convolve_columns(temp, local_height_, width_,
                                                                            vertical_kernel, local_res);

  if (world.rank() == 0) {
    result_.resize(size_);
    int send_start = (base_height + (height_ % world_size)) * width_;
    std::copy(local_res.begin(), local_res.end() - width_, result_.begin());
    for (int proc = 1; proc < world.size(); ++proc) {
      std::vector<uint8_t> buffer((base_height + (proc == (world.size() - 1) ? 1 : 2)) * width_);
      world.recv(proc, 1, buffer);
      std::copy(buffer.begin() + width_, buffer.end() - (proc == world.size() - 1 ? 0 : width_),
                result_.begin() + send_start + (proc - 1) * base_height * width_);
    }
  } else {
    world.send(0, 1, local_res);
  }
  return true;
}

bool zolotareva_a_smoothing_image_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* output_raw = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
    std::copy(result_.begin(), result_.end(), output_raw);
  }
  return true;
}
