#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <limits>

namespace shuravina_o_contrast {

bool ContrastTaskParallel::pre_processing() {
  internal_order_test();

  // Calculate delta only on rank 0
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);  // Broadcast delta to all processes

  // Distribute input data
  if (world.rank() == 0) {
    input_ = std::vector<uint8_t>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());

    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  }
  local_input_ = std::vector<uint8_t>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<uint8_t>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  output_ = std::vector<uint8_t>(delta);
  min_val_ = *std::min_element(local_input_.begin(), local_input_.end());
  max_val_ = *std::max_element(local_input_.begin(), local_input_.end());
  uint8_t global_min;
  uint8_t global_max;
  reduce(world, min_val_, global_min, boost::mpi::minimum<uint8_t>(), 0);
  reduce(world, max_val_, global_max, boost::mpi::maximum<uint8_t>(), 0);
  min_val_ = global_min;
  max_val_ = global_max;
  return true;
}

bool ContrastTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == taskData->inputs_count[0];
  }
  return true;
}

bool ContrastTaskParallel::run() {
  internal_order_test();

  // Calculate delta only on rank 0
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);  // Broadcast delta to all processes

  // Perform contrast enhancement
  if (max_val_ == min_val_) {
    std::fill(output_.begin(), output_.end(), 128);
  } else {
    for (size_t i = 0; i < local_input_.size(); ++i) {
      output_[i] = static_cast<uint8_t>((local_input_[i] - min_val_) * 255.0 / (max_val_ - min_val_));
    }
  }

  // Gather output data
  if (world.rank() == 0) {
    auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
    std::copy(output_.begin(), output_.begin() + delta, tmp_ptr);

    for (int proc = 1; proc < world.size(); proc++) {
      world.recv(proc, 0, tmp_ptr + proc * delta, delta);
    }
  } else {
    world.send(0, 0, output_.data(), delta);
  }

  return true;
}

bool ContrastTaskParallel::post_processing() {
  internal_order_test();
  return true;
}

}  // namespace shuravina_o_contrast