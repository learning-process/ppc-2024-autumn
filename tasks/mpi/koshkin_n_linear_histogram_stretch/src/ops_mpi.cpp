#include "mpi/koshkin_n_linear_histogram_stretch/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> koshkin_n_linear_histogram_stretch_mpi::getRandomImage(int sz) {
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

bool koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  int size = taskData->inputs_count[0];
  image_input = std::vector<int>(size);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + size, image_input.begin());
  image_output = {};
  return true;
}

bool koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  int size = taskData->inputs_count[0];
  if (size % 3 != 0) return false;

  for (int i = 0; i < size; ++i) {
    int value = reinterpret_cast<int*>(taskData->inputs[0])[i];
    if (value < 0 || value > 255) {
      return false;
    }
  }

  return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
          (!taskData->inputs_count.empty() && taskData->inputs_count[0] != 0) &&
          (!taskData->outputs_count.empty() && taskData->outputs_count[0] != 0));
}

bool koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  int size = image_input.size();
  image_output.resize(size);
  int Imin = 255, Imax = 0;

  std::vector<int> I(size / 3);
  for (int i = 0, k = 0; i < size; i += 3, ++k) {
    int R = image_input[i];
    int G = image_input[i + 1];
    int B = image_input[i + 2];

    I[k] = static_cast<int>(0.299 * R + 0.587 * G + 0.114 * B);

    if (I[k] < Imin) Imin = I[k];
    if (I[k] > Imax) Imax = I[k];
  }

  if (Imin == Imax) {
    image_output = image_input;
    return true;
  }

  for (int i = 0, k = 0; i < size; i += 3, ++k) {
    int Inew = ((I[k] - Imin) * 255) / (Imax - Imin);

    float coeff = static_cast<float>(Inew) / static_cast<float>(I[k]);

    image_output[i] = std::min(255, static_cast<int>(image_input[i] * coeff));
    image_output[i + 1] = std::min(255, static_cast<int>(image_input[i + 1] * coeff));
    image_output[i + 2] = std::min(255, static_cast<int>(image_input[i + 2] * coeff));
  }

  return true;
}

bool koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  auto* output = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(image_output.begin(), image_output.end(), output);
  return true;
}

bool koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int size = taskData->inputs_count[0];
    image_input = std::vector<int>(size);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + size, image_input.begin());
    image_output = {};
    return true;
  }
  return true;
}

bool koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    int size = taskData->inputs_count[0];
    if (size % 3 != 0) return false;

    for (int i = 0; i < size; ++i) {
      int value = reinterpret_cast<int*>(taskData->inputs[0])[i];
      if (value < 0 || value > 255) {
        return false;
      }
    }

    return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
            (!taskData->inputs_count.empty() && taskData->inputs_count[0] != 0) &&
            (!taskData->outputs_count.empty() && taskData->outputs_count[0] != 0));
  }
  return true;
}

bool koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int size = 0;
  if (world.rank() == 0) {
    size = image_input.size();
  }

  broadcast(world, size, 0);

  int pixels_per_process = num_pixels / world.size();
  int extra_pixels = num_pixels % world.size();

  // Ћокальное количество пикселей
  int local_pixels = pixels_per_process + (world.rank() < extra_pixels ? 1 : 0);

  std::vector<int> displacements(world.size(), 0);
  std::vector<int> send_counts(world.size(), 0);

  if (world.rank() == 0) {
    for (int proc = 0; proc < world.size(); ++proc) {
      send_counts[proc] = (pixels_per_process + (proc < extra_pixels ? 1 : 0)) * 3;
      if (proc > 0) {
        displacements[proc] = displacements[proc - 1] + send_counts[proc - 1];
      }
    }
  }

  broadcast(world, send_counts.data(), send_counts.size(), 0);
  broadcast(world, displacements.data(), displacements.size(), 0);

  std::vector<int> local_input(local_pixels * 3);
  boost::mpi::scatterv(world, image_input.data(), send_counts, displacements, local_input.data(), local_pixels * 3, 0);

  int local_Imin = 255, local_Imax = 0;
  std::vector<int> local_I(local_pixels);
  for (int i = 0, k = 0; i < local_pixels * 3; i += 3, ++k) {
    int R = local_input[i];
    int G = local_input[i + 1];
    int B = local_input[i + 2];

    local_I[k] = static_cast<int>(0.299 * R + 0.587 * G + 0.114 * B);
    local_Imin = std::min(local_Imin, local_I[k]);
    local_Imax = std::max(local_Imax, local_I[k]);
  }

  int global_Imin, global_Imax;
  boost::mpi::all_reduce(world, local_Imin, global_Imin, boost::mpi::minimum<int>());
  boost::mpi::all_reduce(world, local_Imax, global_Imax, boost::mpi::maximum<int>());

  if (global_Imin == global_Imax) {
    if (world.rank() == 0) {
      image_output = image_input;
    }
    return true;
  }

  std::vector<int> local_output(local_pixels * 3);
  for (int i = 0, k = 0; i < local_pixels * 3; i += 3, ++k) {
    int Inew = ((local_I[k] - global_Imin) * 255) / (global_Imax - global_Imin);
    float coeff = static_cast<float>(Inew) / static_cast<float>(local_I[k]);

    local_output[i] = std::min(255, static_cast<int>(local_input[i] * coeff));
    local_output[i + 1] = std::min(255, static_cast<int>(local_input[i + 1] * coeff));
    local_output[i + 2] = std::min(255, static_cast<int>(local_input[i + 2] * coeff));
  }

  if (world.rank() == 0) {
    image_output.resize(size);
  }

  boost::mpi::gatherv(world, local_output.data(), local_pixels * 3, image_output.data(), send_counts, displacements, 0);

  return true;
}

bool koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* output = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(image_output.begin(), image_output.end(), output);
  }
  return true;
}