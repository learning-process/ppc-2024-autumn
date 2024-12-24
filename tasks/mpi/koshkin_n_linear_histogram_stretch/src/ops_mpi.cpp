#include "mpi/koshkin_n_linear_histogram_stretch/include/ops_mpi.hpp"

#include <string>
#include <thread>
#include <vector>

bool koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  int size = taskData->inputs_count[0];
  image_input = std::vector<int>(size);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + size, image_input.begin());

  int pixel_count = size / 3;
  I.resize(pixel_count);
  for (int i = 0, k = 0; i < size; i += 3, ++k) {
    int r = image_input[i];
    int g = image_input[i + 1];
    int b = image_input[i + 2];

    I[k] = static_cast<int>(0.299 * static_cast<double>(r) + 0.587 * static_cast<double>(g) +
                            0.114 * static_cast<double>(b));
  }

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
  int Imin = 255;
  int Imax = 0;

  for (int intensity : I) {
    Imin = std::min(Imin, intensity);
    Imax = std::max(Imax, intensity);
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

  int num_pixels = 0;
  int pixels_per_process = 0;
  int extra_pixels = 0;
  if (world.rank() == 0) {
    num_pixels = size / 3;
    pixels_per_process = num_pixels / world.size();
    extra_pixels = num_pixels % world.size();
  }

  broadcast(world, num_pixels, 0);
  broadcast(world, pixels_per_process, 0);
  broadcast(world, extra_pixels, 0);

  int local_pixels = pixels_per_process + (world.rank() < extra_pixels ? 1 : 0);

  std::vector<int> offset(world.size(), 0);
  std::vector<int> send_counts(world.size(), 0);

  if (world.rank() == 0) {
    for (int proc = 0; proc < world.size(); ++proc) {
      send_counts[proc] = (pixels_per_process + (proc < extra_pixels ? 1 : 0)) * 3;
      if (proc > 0) {
        offset[proc] = offset[proc - 1] + send_counts[proc - 1];
      }
    }
  }

  broadcast(world, send_counts.data(), send_counts.size(), 0);
  broadcast(world, offset.data(), offset.size(), 0);

  std::vector<int> local_input(local_pixels * 3);
  boost::mpi::scatterv(world, image_input.data(), send_counts, offset, local_input.data(), local_pixels * 3, 0);

  int local_Imin = 255;
  int local_Imax = 0;
  std::vector<int> local_I(local_pixels);
  for (int i = 0, k = 0; i < local_pixels * 3; i += 3, ++k) {
    int R = local_input[i];
    int G = local_input[i + 1];
    int B = local_input[i + 2];

    local_I[k] = static_cast<int>(0.299 * static_cast<double>(R) + 0.587 * static_cast<double>(G) +
                                  0.114 * static_cast<double>(B));
    local_Imin = std::min(local_Imin, local_I[k]);
    local_Imax = std::max(local_Imax, local_I[k]);
  }

  int global_Imin;
  int global_Imax;
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

  boost::mpi::gatherv(world, local_output.data(), local_pixels * 3, image_output.data(), send_counts, offset, 0);

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