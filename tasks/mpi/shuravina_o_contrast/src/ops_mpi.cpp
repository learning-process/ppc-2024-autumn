#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <limits>

namespace shuravina_o_contrast {

bool ContrastTaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    // Init vectors
    input_ = std::vector<uint8_t>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    output_ = std::vector<uint8_t>(taskData->inputs_count[0]);
  }
  return true;
}

bool ContrastTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[0] <= 0) {
      return false;
    }
  }
  return true;
}

bool ContrastTaskParallel::run() {
  internal_order_test();
  unsigned int num_processes = world.size();
  std::vector<int> send_counts(num_processes, 0);
  std::vector<int> displacements(num_processes, 0);
  std::vector<int> output_displacements(num_processes, 0);
  std::vector<int> output_counts(num_processes, 0);

  if (world.rank() == 0) {
    for (size_t i = 0; i < num_processes; i++) {
      send_counts[i] = input_.size() / world.size();
      if (i == (size_t)num_processes - 1) {
        send_counts[i] = input_.size() - i * (input_.size() / world.size());
      }
      output_counts[i] = send_counts[i];
    }

    for (size_t i = 1; i < num_processes; i++) {
      displacements[i] = displacements[i - 1] + send_counts[i - 1];
      output_displacements[i] = displacements[i];
    }
  }

  for (size_t i = 0; i < num_processes; i++) {
    broadcast(world, send_counts[i], 0);
    broadcast(world, displacements[i], 0);
    broadcast(world, output_counts[i], 0);
    broadcast(world, output_displacements[i], 0);
  }

  std::vector<uint8_t> local_input_(send_counts[world.rank()]);
  boost::mpi::scatterv(world, input_.data(), send_counts, displacements, local_input_.data(), local_input_.size(), 0);

  uint8_t local_min_val = *std::min_element(local_input_.begin(), local_input_.end());
  uint8_t local_max_val = *std::max_element(local_input_.begin(), local_input_.end());

  uint8_t global_min_val, global_max_val;
  reduce(world, local_min_val, global_min_val, boost::mpi::minimum<uint8_t>(), 0);
  reduce(world, local_max_val, global_max_val, boost::mpi::maximum<uint8_t>(), 0);

  broadcast(world, global_min_val, 0);
  broadcast(world, global_max_val, 0);

  std::vector<uint8_t> local_output_(local_input_.size());
  if (global_max_val == global_min_val) {
    std::fill(local_output_.begin(), local_output_.end(), 128);
  } else {
    for (size_t i = 0; i < local_input_.size(); ++i) {
      local_output_[i] =
          static_cast<uint8_t>((local_input_[i] - global_min_val) * 255.0 / (global_max_val - global_min_val));
    }
  }

  boost::mpi::gatherv(world, local_output_.data(), local_output_.size(), output_.data(), output_counts,
                      output_displacements, 0);

  return true;
}

bool ContrastTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<std::vector<uint8_t>*>(taskData->outputs[0]) = output_;
  }
  return true;
}

}  // namespace shuravina_o_contrast