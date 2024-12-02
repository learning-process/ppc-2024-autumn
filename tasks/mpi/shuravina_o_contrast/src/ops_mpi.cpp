#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <cmath>
#include <iostream>

bool shuravina_o_contrast::ContrastParallel::pre_processing() {
  if (world.rank() == 0) {
    size_t data_size = taskData->inputs_count[0];
    input_.resize(data_size);
    auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + data_size, input_.begin());
  }

  return true;
}

bool shuravina_o_contrast::ContrastParallel::validation() {
  if (world.rank() == 0) {
    return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
            (taskData->outputs_count[0] == taskData->inputs_count[0]));
  }
  return true;
}

bool shuravina_o_contrast::ContrastParallel::run() {
  int total_size = taskData->inputs_count[0];

  int chunk_size = total_size / world.size();
  int remainder = total_size % world.size();

  int local_size = chunk_size + (world.rank() < remainder ? 1 : 0);
  local_input_.resize(local_size);

  std::vector<int> counts(world.size(), chunk_size);
  for (int i = 0; i < remainder; ++i) {
    counts[i]++;
  }
  std::vector<int> displs(world.size(), 0);
  for (int i = 1; i < world.size(); ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  boost::mpi::scatterv(world, input_.data(), counts, displs, local_input_.data(), local_size, 0);

  uint8_t p_min_local_ = *std::min_element(local_input_.begin(), local_input_.end());
  uint8_t p_max_local_ = *std::max_element(local_input_.begin(), local_input_.end());

  boost::mpi::all_reduce(world, p_min_local_, p_min_global_, boost::mpi::minimum<uint8_t>());
  boost::mpi::all_reduce(world, p_max_local_, p_max_global_, boost::mpi::maximum<uint8_t>());

  if (p_max_global_ == p_min_global_) {
    std::fill(local_input_.begin(), local_input_.end(), 0);
  } else {
    for (auto& pixel : local_input_) {
      pixel = static_cast<uint8_t>(static_cast<double>(pixel - p_min_global_) * 255 / (p_max_global_ - p_min_global_));
    }
  }

  if (world.rank() == 0) {
    output_.resize(total_size, 0);
  }

  boost::mpi::gatherv(world, local_input_.data(), local_input_.size(), output_.data(), counts, displs, 0);

  return true;
}

bool shuravina_o_contrast::ContrastParallel::post_processing() {
  if (world.rank() == 0) {
    std::copy(output_.begin(), output_.end(), reinterpret_cast<uint8_t*>(taskData->outputs[0]));
  }
  return true;
}