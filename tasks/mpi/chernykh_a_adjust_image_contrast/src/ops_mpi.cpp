#include "mpi/chernykh_a_adjust_image_contrast/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

namespace chernykh_a_adjust_image_contrast_mpi {

bool SequentialTask::validation() {
  internal_order_test();
  return contrast_factor >= 0.0 && contrast_factor <= 2.0 &&       // Contrast factor within [0.0, 2.0]
         taskData->inputs_count[0] > 0 &&                          // Input image must have at least 1 pixel
         taskData->inputs_count[0] == taskData->outputs_count[0];  // Input and output image sizes match
}

bool SequentialTask::pre_processing() {
  internal_order_test();
  auto* input_ptr = reinterpret_cast<Pixel*>(taskData->inputs[0]);
  auto input_size = taskData->inputs_count[0];
  auto result_size = taskData->outputs_count[0];
  input.assign(input_ptr, input_ptr + input_size);
  result.resize(result_size);
  return true;
}

bool SequentialTask::run() {
  internal_order_test();
  std::transform(input.begin(), input.end(), result.begin(),
                 [this](const Pixel& p) { return p.with_contrast(contrast_factor); });
  return true;
}

bool SequentialTask::post_processing() {
  internal_order_test();
  auto* result_ptr = reinterpret_cast<Pixel*>(taskData->outputs[0]);
  std::copy(result.begin(), result.end(), result_ptr);
  return true;
}

bool ParallelTask::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return contrast_factor >= 0.0 && contrast_factor <= 2.0 &&       // Contrast factor within [0.0, 2.0]
           taskData->inputs_count[0] > 0 &&                          // Input image must have at least 1 pixel
           taskData->inputs_count[0] == taskData->outputs_count[0];  // Input and output image sizes match
  }
  return true;
}

bool ParallelTask::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* input_ptr = reinterpret_cast<Pixel*>(taskData->inputs[0]);
    auto input_size = taskData->inputs_count[0];
    auto result_size = taskData->outputs_count[0];
    input.assign(input_ptr, input_ptr + input_size);
    result.resize(result_size);
  }
  return true;
}

bool ParallelTask::run() {
  internal_order_test();
  auto local_input = std::vector<Pixel>();
  auto local_result = std::vector<Pixel>();
  auto sizes = std::vector<int>();

  if (world.rank() == 0) {
    auto input_size = int(input.size());
    auto num_process = std::min(world.size(), input_size);
    auto size = input_size / num_process;
    auto remainder = input_size % num_process;

    sizes.resize(num_process, size);
    std::fill(sizes.begin() + 1, sizes.begin() + 1 + remainder, size + 1);
  }

  boost::mpi::broadcast(world, sizes, 0);
  local_input.resize(sizes[world.rank()]);
  boost::mpi::scatterv(world, input, sizes, local_input.data(), 0);

  if (!local_input.empty()) {
    local_result.resize(local_input.size());
    std::transform(local_input.begin(), local_input.end(), local_result.begin(),
                   [this](const Pixel& p) { return p.with_contrast(contrast_factor); });
  }

  boost::mpi::gatherv(world, local_result, result.data(), sizes, 0);
  return true;
}

bool ParallelTask::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* result_ptr = reinterpret_cast<Pixel*>(taskData->outputs[0]);
    std::copy(result.begin(), result.end(), result_ptr);
  }
  return true;
}

}  // namespace chernykh_a_adjust_image_contrast_mpi
