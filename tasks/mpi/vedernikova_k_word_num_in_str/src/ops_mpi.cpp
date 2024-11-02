#include "../include/ops_mpi.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool vedernikova_k_word_num_in_str_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == 1;
}

bool vedernikova_k_word_num_in_str_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  input_.assign(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);

  res_ = 0;

  return true;
}

bool vedernikova_k_word_num_in_str_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  bool is_space = input_[0] == ' ';
  for (const char c : input_) {
    if (c == ' ') {
      if (!is_space) {
        res_++;
      }
      is_space = true;
      continue;
    }
    is_space = false;
  }
  res_ += (is_space || input_.empty()) ? 0 : 1;

  return true;
}

bool vedernikova_k_word_num_in_str_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  *reinterpret_cast<size_t*>(taskData->outputs[0]) = res_;

  return true;
}

//

bool vedernikova_k_word_num_in_str_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  return world.rank() != 0 || taskData->outputs_count[0] == 1;
}

bool vedernikova_k_word_num_in_str_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    input_.assign(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  }
  res_ = 0;

  return true;
}

bool vedernikova_k_word_num_in_str_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  size_t load{};
  if (world.rank() == 0) {
    load = input_.size() / world.size();
  }
  boost::mpi::broadcast(world, load, 0);
  const size_t underlap = 1;

  std::string local_input;
  if (world.rank() == 0) {
    const size_t extra_load = input_.size() % world.size();

    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * load + extra_load - underlap, load + underlap);
    }

    local_input.assign(input_, 0, load + extra_load);
  } else {
    local_input.resize(load + underlap);
    world.recv(0, 0, local_input.data(), load + underlap);
  }

  size_t local_res = 0;
  auto it = local_input.begin();
  if (world.rank() != 0 && it != local_input.end()) {
    bool skip_is_space = *it == ' ';
    for (; it != local_input.end() && (skip_is_space == (*it == ' ')); ++it)
      ;
  }
  const bool ended = it == local_input.end();
  bool is_space = !ended && *it == ' ';

  for (; it != local_input.end(); ++it) {
    if (*it == ' ') {
      if (!is_space) {
        local_res++;
      }
      is_space = true;
      continue;
    }
    is_space = false;
  }
  local_res += (ended || is_space || local_input.empty()) ? 0 : 1;

  boost::mpi::reduce(world, local_res, res_, std::plus(), 0);

  return true;
}

bool vedernikova_k_word_num_in_str_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<size_t*>(taskData->outputs[0]) = res_;
  }
  return true;
}
