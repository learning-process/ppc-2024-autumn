// Copyright 2024 Nesterov Alexander
#include "mpi/kazunin_n_count_freq_a_char_in_string/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cstring>

bool kazunin_n_count_freq_a_char_in_string_mpi::CharFreqCounterMPISequential::pre_processing() {
  internal_order_test();
  input_string_.assign(reinterpret_cast<char*>(taskData->inputs[0]),
                       reinterpret_cast<char*>(taskData->inputs[0]) + taskData->inputs_count[0]);
  character_to_count_ = *reinterpret_cast<char*>(taskData->inputs[1]);
  count_result_ = 0;
  return true;
}

bool kazunin_n_count_freq_a_char_in_string_mpi::CharFreqCounterMPISequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool kazunin_n_count_freq_a_char_in_string_mpi::CharFreqCounterMPISequential::run() {
  internal_order_test();
  count_result_ = std::count(input_string_.begin(), input_string_.end(), character_to_count_);
  return true;
}

bool kazunin_n_count_freq_a_char_in_string_mpi::CharFreqCounterMPISequential::post_processing() {
  *reinterpret_cast<size_t*>(taskData->outputs[0]) = count_result_;
  return true;
}

bool kazunin_n_count_freq_a_char_in_string_mpi::CharFreqCounterMPIParallel::pre_processing() {
  internal_order_test();
  if (global.rank() == 0) {
    character_to_count_ = *reinterpret_cast<char*>(taskData->inputs[1]);
  }
  return true;
}

bool kazunin_n_count_freq_a_char_in_string_mpi::CharFreqCounterMPIParallel::validation() {
  internal_order_test();
  return global.rank() != 0 || taskData->outputs_count[0] == 1;
}

bool kazunin_n_count_freq_a_char_in_string_mpi::CharFreqCounterMPIParallel::run() {
  internal_order_test();

  int my_rank = global.rank();
  auto world_size = global.size();
  int n = 0;

  if (my_rank == 0) {
    n = taskData->inputs_count[0];
    input_string_.assign(reinterpret_cast<char*>(taskData->inputs[0]),
                         reinterpret_cast<char*>(taskData->inputs[0]) + n);
  }

  boost::mpi::broadcast(global, n, 0);
  boost::mpi::broadcast(global, character_to_count_, 0);

  auto base_segment_size = n / world_size;
  auto extra = n % world_size;
  std::vector<int> send_counts(world_size, base_segment_size);
  std::vector<int> displacements(world_size, 0);

  for (auto i = 0; i < world_size; ++i) {
    if (i < extra) {
      ++send_counts[i];
    }
    if (i > 0) {
      displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }
  }

  local_segment_.resize(send_counts[my_rank]);
  if (my_rank == 0) {
    boost::mpi::scatterv(global, input_string_.data(), send_counts, displacements, local_segment_.data(),
                         send_counts[my_rank], 0);
  } else {
    std::vector<char> empty_buffer(0);
    boost::mpi::scatterv(global, empty_buffer.data(), send_counts, displacements, local_segment_.data(),
                         send_counts[my_rank], 0);
  }

  local_count_ = std::count(local_segment_.begin(), local_segment_.end(), character_to_count_);

  boost::mpi::reduce(global, local_count_, total_count_, std::plus<>(), 0);
  return true;
}

bool kazunin_n_count_freq_a_char_in_string_mpi::CharFreqCounterMPIParallel::post_processing() {
  internal_order_test();

  if (global.rank() == 0) {
    *reinterpret_cast<size_t*>(taskData->outputs[0]) = total_count_;
  }
  return true;
}
