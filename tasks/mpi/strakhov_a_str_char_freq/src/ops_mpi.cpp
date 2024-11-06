#include "mpi/strakhov_a_str_char_freq/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool strakhov_a_str_char_freq_mpi::StringCharactersFrequencySequentional::pre_processing() {
  internal_order_test();
  input_ = std::vector<char>(taskData->inputs_count[0]);
  input_.assign(reinterpret_cast<char*>(taskData->inputs[0]),
                reinterpret_cast<char*>(taskData->inputs[0]) + taskData->inputs_count[0]);
  target_ = *reinterpret_cast<char*>(taskData->inputs[1]);
  res = 0;
  return true;
}

bool strakhov_a_str_char_freq_mpi::StringCharactersFrequencySequentional::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool strakhov_a_str_char_freq_mpi::StringCharactersFrequencySequentional::run() {
  internal_order_test();
  res = std::count(input_.begin(), input_.end(), target_);
  return true;
}

bool strakhov_a_str_char_freq_mpi::StringCharactersFrequencySequentional::post_processing() {
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool strakhov_a_str_char_freq_mpi::StringCharactersFrequencyParallel::pre_processing() {
  internal_order_test();
  int world_size = world.size();
  unsigned int n = 0;
  if (world.rank()) {
    n = taskData->inputs_count[0];
    input_ = std::vector<char>(taskData->inputs[0], taskData->inputs[0] + n);
    target_ = *reinterpret_cast<char*>(taskData->inputs[1]);
  }
  boost::mpi::broadcast(world, n, 0);
  boost::mpi::broadcast(world, target_, 0);
  unsigned int substring_size = n / world_size;

  unsigned int overflow_size = n % world_size;
  std::vector<int> send_counts(world_size, substring_size + (overflow_size > 0 ? 1 : 0));
  std::vector<int> displacements(world_size, 0);
  for (unsigned int i = 1; i < static_cast<unsigned int>(world_size); ++i) {
    if (i >= overflow_size) send_counts[i] = substring_size;
    displacements[i] = displacements[i - 1] + send_counts[i - 1];
  }
  local_input_.resize(send_counts[world.rank()]);
  boost::mpi::scatterv(world, input_.data(), send_counts, displacements, local_input_.data(), send_counts[world.rank()],
                       0);

  return true;
}

bool strakhov_a_str_char_freq_mpi::StringCharactersFrequencyParallel::run() {
  internal_order_test();

  local_res = std::count(local_input_.begin(), local_input_.end(), target_);
  boost::mpi::reduce(world, local_res, res, std::plus<>(), 0);

  return true;
}

bool strakhov_a_str_char_freq_mpi::StringCharactersFrequencyParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool strakhov_a_str_char_freq_mpi::StringCharactersFrequencyParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }

  return true;
}