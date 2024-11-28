#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <cmath>
#include <iostream>

bool shuravina_o_contrast::ContrastParallel::pre_processing() {
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  boost::mpi::broadcast(world, delta, 0);

  if (world.rank() == 0) {
    input_ = std::vector<uint8_t>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
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
  return true;
}

bool shuravina_o_contrast::ContrastParallel::validation() {
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == taskData->inputs_count[0];
  }
  return true;
}

bool shuravina_o_contrast::ContrastParallel::run() {
  uint8_t local_min_val = *std::min_element(local_input_.begin(), local_input_.end());
  uint8_t local_max_val = *std::max_element(local_input_.begin(), local_input_.end());

  uint8_t global_min_val, global_max_val;
  boost::mpi::reduce(world, local_min_val, global_min_val, boost::mpi::minimum<uint8_t>(), 0);
  boost::mpi::reduce(world, local_max_val, global_max_val, boost::mpi::maximum<uint8_t>(), 0);

  if (world.rank() == 0) {
    std::cout << "Global min: " << static_cast<int>(global_min_val)
              << ", Global max: " << static_cast<int>(global_max_val) << std::endl;
    if (global_min_val == global_max_val) {
      std::fill(output_.begin(), output_.end(), 255);
    } else {
      for (size_t i = 0; i < local_input_.size(); ++i) {
        output_[i] =
            static_cast<uint8_t>((local_input_[i] - global_min_val) * 255.0 / (global_max_val - global_min_val));
        std::cout << "Output[" << i << "]: " << static_cast<int>(output_[i]) << std::endl;
      }
    }
  }
  return true;
}

bool shuravina_o_contrast::ContrastParallel::post_processing() {
  std::vector<uint8_t> gathered_output;
  if (world.rank() == 0) {
    gathered_output.resize(taskData->outputs_count[0]);
  }

  boost::mpi::gather(world, output_.data(), output_.size(), gathered_output.data(), 0);

  if (world.rank() == 0) {
    auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
    for (size_t i = 0; i < taskData->outputs_count[0]; i++) {
      tmp_ptr[i] = gathered_output[i];
    }
  }
  return true;
}