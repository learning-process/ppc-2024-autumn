// Copyright 2023 Nesterov Alexander
#include "mpi/zaytsev_d_num_of_alternations_signs/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <vector>

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int input_count = taskData->inputs_count[0];
  data_.assign(input_data, input_data + input_count);
  res = 0;
  return true;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  if (data_.size() < 2) {
    res = 0;
    return true;
  }

  for (size_t i = 1; i < data_.size(); ++i) {
    if ((data_[i] >= 0 && data_[i - 1] < 0) || (data_[i] < 0 && data_[i - 1] >= 0)) {
      res++;
    }
  }

  return true;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  }

  res = 0;
  return true;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  unsigned int delta = 0;
  unsigned int remainder = 0;

  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    remainder = taskData->inputs_count[0] % world.size();
  }

  broadcast(world, delta, 0);
  broadcast(world, remainder, 0);

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      int send_count = delta + (proc == world.size() - 1 ? remainder : 0);
      world.send(proc, 0, input_.data() + proc * (int)delta, send_count);
    }
  }

  int local_size = (int)delta + (world.rank() == world.size() - 1 ? remainder : 0);
  local_input_ = std::vector<int>(local_size);

  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + local_size);
  } else {
    world.recv(0, 0, local_input_.data(), local_size);
  }

  int local_count = 0;

  if (local_input_.size() > 1) {
    for (size_t i = 1; i < local_input_.size(); i++) {
      if ((local_input_[i - 1] >= 0 && local_input_[i] < 0) || (local_input_[i - 1] < 0 && local_input_[i] >= 0)) {
        local_count++;
      }
    }
  }

  int prev_value = 0;
  if (world.rank() > 0) {
    world.recv(world.rank() - 1, 0, &prev_value, 1);
    if (!local_input_.empty() &&
        ((prev_value >= 0 && local_input_[0] < 0) || (prev_value < 0 && local_input_[0] >= 0))) {
      local_count++;
    }
  }

  int last_value = local_input_.empty() ? 0 : local_input_.back();
  if (world.rank() < world.size() - 1) {
    world.send(world.rank() + 1, 0, &last_value, 1);
  }

  boost::mpi::reduce(world, local_count, res, std::plus<>(), 0);
  return true;
}

bool zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }

  return true;
}