#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <iostream>

bool shuravina_o_contrast::ContrastTaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    std::cout << "Delta: " << delta << std::endl;
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    input_ = std::vector<uint8_t>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, &input_[proc * delta], delta);
    }
  }
  local_input_ = std::vector<uint8_t>(delta);
  if (world.rank() == 0) {
    local_input_ = std::vector<uint8_t>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, &local_input_[0], delta);
  }
  output_ = std::vector<uint8_t>(delta);
  return true;
}

bool shuravina_o_contrast::ContrastTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == taskData->inputs_count[0];
  }
  return true;
}

bool shuravina_o_contrast::ContrastTaskParallel::run() {
  internal_order_test();
  uint8_t local_min_val = *std::min_element(local_input_.begin(), local_input_.end());
  uint8_t local_max_val = *std::max_element(local_input_.begin(), local_input_.end());

  uint8_t global_min_val = 0, global_max_val = 0;
  reduce(world, local_min_val, global_min_val, boost::mpi::minimum<uint8_t>(), 0);
  reduce(world, local_max_val, global_max_val, boost::mpi::maximum<uint8_t>(), 0);

  if (world.rank() == 0) {
    std::cout << "Local min: " << static_cast<int>(local_min_val) << ", Local max: " << static_cast<int>(local_max_val)
              << std::endl;
    std::cout << "Global min: " << static_cast<int>(global_min_val)
              << ", Global max: " << static_cast<int>(global_max_val) << std::endl;
  }

  if (global_max_val == global_min_val) {
    std::fill(output_.begin(), output_.end(), 0);
  } else {
    for (size_t i = 0; i < local_input_.size(); ++i) {
      output_[i] = static_cast<uint8_t>((local_input_[i] - global_min_val) * 255.0 / (global_max_val - global_min_val));
    }
  }

  return true;
}

bool shuravina_o_contrast::ContrastTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::vector<uint8_t> global_output(taskData->inputs_count[0]);
    gather(world, &output_[0], output_.size(), &global_output[0], 0);
    auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->outputs[0]);
    for (unsigned i = 0; i < taskData->outputs_count[0]; i++) {
      tmp_ptr[i] = global_output[i];
    }
  } else {
    gather(world, &output_[0], output_.size(), 0);
  }
  return true;
}