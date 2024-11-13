#include "mpi/vavilov_v_contrast_enhancement/include/ops_mpi.hpp"

bool vavilov_v_contrast_enhancement_mpi::ContrastEnhancementParallel::pre_processing() {
  internal_order_test();
  int total_size = taskData->inputs_count[0];

  int chunk_size = total_size / world.size();
  int remainder = total_size % world.size();

  if (world.rank() == 0) {
    input_.resize(total_size);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + total_size, input_.begin());
  }

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

  boost::mpi::scatterv(world, input_, counts, displs, local_input_, 0);

  if (world.rank() == 0) {
    output_.resize(total_size, 0);
  }
  return true;
}

bool vavilov_v_contrast_enhancement_mpi::ContrastEnhancementParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return taskData->outputs_count[0] == taskData->inputs_count[0];
  }
  return true;
}

bool vavilov_v_contrast_enhancement_mpi::ContrastEnhancementParallel::run() {
  internal_order_test();
  if (local_input_.empty()) {
    return false;
  }

  p_min_local_ = *std::min_element(local_input_.begin(), local_input_.end());
  p_max_local_ = *std::max_element(local_input_.begin(), local_input_.end());

  boost::mpi::all_reduce(world, p_min_local_, p_min_global_, [](int a, int b) { return std::min(a, b); });
  boost::mpi::all_reduce(world, p_max_local_, p_max_global_, [](int a, int b) { return std::max(a, b); });

  if (p_max_global_ == p_min_global_) {
    std::fill(local_input_.begin(), local_input_.end(), 0);
  } else {
    for (auto& pixel : local_input_) {
      pixel = static_cast<int>(static_cast<double>(pixel - p_min_global_) * 255 / (p_max_global_ - p_min_global_));
    }
  }

  boost::mpi::gatherv(world, local_input_, output_, 0);
  return true;
}

bool vavilov_v_contrast_enhancement_mpi::ContrastEnhancementParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(output_.begin(), output_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
