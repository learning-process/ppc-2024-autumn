#include "mpi/vasenkov_a_char_freq/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool vasenkov_a_char_freq_mpi::CharFrequencySequential::pre_processing() {
  internal_order_test();

  str_input_ = std::vector<char>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    str_input_[i] = tmp_ptr[i];
  }

  target_char_ = *reinterpret_cast<char*>(taskData->inputs[1]);
  res = 0;
  return true;
}

bool vasenkov_a_char_freq_mpi::CharFrequencySequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool vasenkov_a_char_freq_mpi::CharFrequencySequential::run() {
  internal_order_test();

  res = std::count(str_input_.begin(), str_input_.end(), target_char_);
  return true;
}

bool vasenkov_a_char_freq_mpi::CharFrequencySequential::post_processing() {
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool vasenkov_a_char_freq_mpi::CharFrequencyParallel::pre_processing() {
  internal_order_test();
  int myid = world.rank();
  int world_size = world.size();
  unsigned int n = 0;

    if (myid == 0) {
      n = taskData->inputs_count[0];
      str_input_ = std::vector<char>(taskData->inputs[0], taskData->inputs[0] + n);
      target_char_ = *reinterpret_cast<char*>(taskData->inputs[1]);
    }

  boost::mpi::broadcast(world, n, 0);
  boost::mpi::broadcast(world, target_char_, 0);

  unsigned int vec_send_size = n / world_size;
  unsigned int overflow_size = n % world_size;

  std::vector<int> send_counts(world_size, vec_send_size + (overflow_size > 0 ? 1 : 0));
  std::vector<int> displs(world_size, 0);

  for (unsigned int i = 1; i < static_cast<unsigned int>(world_size); ++i) {
    if (i >= overflow_size) send_counts[i] = vec_send_size;
    displs[i] = displs[i - 1] + send_counts[i - 1];
  }

  local_input_.resize(send_counts[myid]);
  boost::mpi::scatterv(world, str_input_.data(), send_counts, displs, local_input_.data(), send_counts[myid], 0);
  local_res = 0;
  res = 0;
  return true;
}

bool vasenkov_a_char_freq_mpi::CharFrequencyParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool vasenkov_a_char_freq_mpi::CharFrequencyParallel::run() {
  internal_order_test();
  local_res = std::count(local_input_.begin(), local_input_.end(), target_char_);

  boost::mpi::reduce(world, local_res, res, std::plus<>(), 0);
  return true;
}

bool vasenkov_a_char_freq_mpi::CharFrequencyParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }

  return true;
}
