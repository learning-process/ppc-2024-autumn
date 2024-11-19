#include "mpi/gnitienko_k_contrast_enhancement/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

std::vector<int> gnitienko_k_contrast_enhancement_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 255;
  }
  return vec;
}

bool gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq::is_grayscale() const { return image.size() % 3 != 0; }

bool gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq::pre_processing() {
  internal_order_test();
  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  contrast_factor = *reinterpret_cast<double*>(taskData->inputs[1]);
  size_t input_size = taskData->inputs_count[0];
  image.resize(input_size, 0);
  res.resize(input_size, 0);
  image.assign(input_data, input_data + input_size);

  return true;
}

bool gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == taskData->outputs_count[0] && taskData->inputs_count[0] >= 0;
}

bool gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq::run() {
  internal_order_test();
  if (is_grayscale()) {
    for (size_t i = 0; i < image.size(); ++i) {
      res[i] = std::clamp(static_cast<int>((image[i] - 128) * contrast_factor + 128), 0, 255);
    }
  } else {
    for (size_t i = 0; i < image.size(); i += 3) {
      res[i] = std::clamp(static_cast<int>((image[i] - 128) * contrast_factor + 128), 0, 255);
      res[i + 1] = std::clamp(static_cast<int>((image[i + 1] - 128) * contrast_factor + 128), 0, 255);
      res[i + 2] = std::clamp(static_cast<int>((image[i + 2] - 128) * contrast_factor + 128), 0, 255);
    }
  }
  return true;
}

bool gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < res.size(); ++i) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

// Parallel

bool gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
    contrast_factor = *reinterpret_cast<double*>(taskData->inputs[1]);
    img_size = taskData->inputs_count[0];
    image.resize(img_size);
    image.assign(input_data, input_data + img_size);
  }
  return true;
}

bool gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->inputs_count[0] == taskData->outputs_count[0] && taskData->inputs_count[0] >= 0;
  }
  return true;
}

bool gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI::run() {
  internal_order_test();
  boost::mpi::broadcast(world, contrast_factor, 0);
  boost::mpi::broadcast(world, img_size, 0);
  image.resize(img_size);
  boost::mpi::broadcast(world, image.data(), img_size, 0);
  int count_of_proc = world.size();
  int rank = world.rank();
  int pixels_per_process = 0;
  std::vector<int> local_res;
  if (is_grayscale()) {
    pixels_per_process = img_size / count_of_proc;
    if (img_size % count_of_proc != 0) pixels_per_process += 1;

    local_res.resize(pixels_per_process);
    for (int i = rank * pixels_per_process; i < std::min(img_size, (rank + 1) * pixels_per_process); ++i) {
      int local_index = i - rank * pixels_per_process;
      local_res[local_index] = std::clamp(static_cast<int>((image[i] - 128) * contrast_factor + 128), 0, 255);
    }
  } else {
    pixels_per_process = (img_size / 3) / count_of_proc;
    if (img_size % (3 * count_of_proc) != 0) {
      pixels_per_process += 1;
    }

    local_res.resize(pixels_per_process * 3);
    for (int i = rank * pixels_per_process * 3; i < std::min(img_size, (rank + 1) * pixels_per_process * 3);
         i += 3) {
      int local_index = i - rank * pixels_per_process * 3;
      local_res[local_index] = std::clamp(static_cast<int>((image[i] - 128) * contrast_factor + 128), 0, 255);
      local_res[local_index + 1] = std::clamp(static_cast<int>((image[i + 1] - 128) * contrast_factor + 128), 0, 255);
      local_res[local_index + 2] = std::clamp(static_cast<int>((image[i + 2] - 128) * contrast_factor + 128), 0, 255);
    }
  }

  if (rank == 0) {
    std::vector<int> image_res(img_size + world.size() * pixels_per_process * (is_grayscale() ? 1 : 3));
    std::vector<int> sizes(world.size(), pixels_per_process * (is_grayscale() ? 1 : 3));
    boost::mpi::gatherv(world, local_res.data(), pixels_per_process * (is_grayscale() ? 1 : 3), image_res.data(), sizes,
                        0);
    image_res.resize(img_size);
    res = image_res;
  } else {
    boost::mpi::gatherv(world, local_res.data(), pixels_per_process * (is_grayscale() ? 1 : 3), 0);
  }

  return true;
}

bool gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < res.size(); ++i) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}

bool gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI::is_grayscale() const { return image.size() % 3 != 0; }
