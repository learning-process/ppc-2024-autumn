#include "../include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/utility.hpp>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>

int16_t conv_kernel3(const std::array<std::array<int8_t, 3>, 3>& kernel, const std::vector<uint8_t>& img, size_t i,
                     size_t j, size_t height) {
  // clang-format off
  return kernel[0][0] * img[(i - 1) * height + (j - 1)]
       + kernel[0][1] * img[(i - 1) * height + j]
       + kernel[0][2] * img[(i - 1) * height + (j + 1)]
       + kernel[1][0] * img[i * height + (j - 1)]
       + kernel[1][1] * img[i * height + j]
       + kernel[1][2] * img[i * height + (j + 1)]
       + kernel[2][0] * img[(i + 1) * height + (j - 1)]
       + kernel[2][1] * img[(i + 1) * height + j]
       + kernel[2][2] * img[(i + 1) * height + (j + 1)];
  // clang-format on
}

bool koshkin_m_sobel_mpi::TestTaskSequential::pre_processing() {
  internal_order_test();

  imgsize = {taskData->inputs_count[0], taskData->inputs_count[1]};
  auto& [width, height] = imgsize;
  const int padding = 2;
  image.resize((width + padding) * (height + padding));
  resimg.resize(width * height, 0);

  const auto* in = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
  for (size_t row = 0; row < height; row++) {
    std::copy(in + (row * width), in + ((row + 1) * width), image.begin() + ((row + 1) * (width + padding) + 1));
  }

  return true;
}

bool koshkin_m_sobel_mpi::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
         taskData->outputs_count[0] == taskData->inputs_count[0] &&
         taskData->outputs_count[1] == taskData->inputs_count[1];
}

bool koshkin_m_sobel_mpi::TestTaskSequential::run() {
  internal_order_test();

  const auto [width, height] = imgsize;

  for (size_t i = 1; i < (width + 2) - 1; i++) {
    for (size_t j = 1; j < (height + 2) - 1; j++) {
      const auto accX = conv_kernel3(SOBEL_KERNEL_X, image, i, j, (height + 2));
      const auto accY = conv_kernel3(SOBEL_KERNEL_Y, image, i, j, (height + 2));
      resimg[(i - 1) * height + (j - 1)] = std::clamp(std::sqrt(std::pow(accX, 2) + std::pow(accY, 2)), 0., 255.);
    }
  }

  return true;
}

bool koshkin_m_sobel_mpi::TestTaskSequential::post_processing() {
  internal_order_test();
  std::copy(resimg.begin(), resimg.end(), reinterpret_cast<uint8_t*>(taskData->outputs[0]));
  return true;
}

bool koshkin_m_sobel_mpi::TestTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    imgsize = {taskData->inputs_count[0], taskData->inputs_count[1]};
    auto& [width, height] = imgsize;

    image.resize((width + 2) * (height + 2));
    resimg.resize(width * height, 0);

    const auto* in = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    for (size_t row = 0; row < height; row++) {
      std::copy(in + (row * width), in + ((row + 1) * width), image.begin() + ((row + 1) * (width + 2) + 1));
    }
  }

  return true;
}

bool koshkin_m_sobel_mpi::TestTaskParallel::validation() {
  internal_order_test();
  return world.rank() != 0 || (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 &&
                               taskData->outputs_count[0] == taskData->inputs_count[0] &&
                               taskData->outputs_count[1] == taskData->inputs_count[1]);
}

bool koshkin_m_sobel_mpi::TestTaskParallel::run() {
  internal_order_test();

  const int pad = 2;

  boost::mpi::broadcast(world, imgsize, 0);
  const auto [width, height] = imgsize;
  const auto [pwidth, pheight] = std::make_pair(width + pad, height + pad);

  const int perproc = width / world.size();
  const int leftover = width % world.size();

  std::vector<int> sendcounts(world.size(), 0);
  std::vector<int> senddispls(world.size(), 0);
  std::vector<int> recvcounts(world.size(), 0);
  std::vector<int> recvdispls(world.size(), 0);

  for (int i = 0; i < world.size(); i++) {
    const int extra = i < leftover ? 1 : 0;
    sendcounts[i] = (perproc + extra + pad) * pheight;
    recvcounts[i] = (perproc + extra) * height;
  }
  for (int i = 1; i < world.size(); i++) {
    senddispls[i] = senddispls[i - 1] + sendcounts[i - 1] - pheight * pad;
    recvdispls[i] = recvdispls[i - 1] + recvcounts[i - 1];
  }

  std::vector<uint8_t> locimg(sendcounts[world.rank()]);
  boost::mpi::scatterv(world, image, sendcounts, senddispls, locimg.data(), locimg.size(), 0);

  const int actw = sendcounts[world.rank()] / pheight;
  std::vector<uint8_t> locres(recvcounts[world.rank()]);
  for (int i = 1; i < actw - 1; i++) {
    for (size_t j = 1; j < (height + 2) - 1; j++) {
      const auto accX = conv_kernel3(SOBEL_KERNEL_X, locimg, i, j, (height + 2));
      const auto accY = conv_kernel3(SOBEL_KERNEL_Y, locimg, i, j, (height + 2));
      locres[(i - 1) * height + (j - 1)] = std::clamp(std::sqrt(std::pow(accX, 2) + std::pow(accY, 2)), 0., 255.);
    }
  }

  boost::mpi::gatherv(world, locres, resimg.data(), recvcounts, recvdispls, 0);

  return true;
}

bool koshkin_m_sobel_mpi::TestTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(resimg.begin(), resimg.end(), reinterpret_cast<uint8_t*>(taskData->outputs[0]));
  }
  return true;
}