#include "mpi/muradov_m_count_alpha_chars/include/ops_mpi.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskSequential::pre_processing() {
  internal_order_test();

  input_str_ = std::vector<char>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_str_[i] = tmp_ptr[i];
  }

  alpha_count_ = 0;
  return true;
}

bool muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskSequential::run() {
  internal_order_test();

  alpha_count_ = std::count_if(input_str_.begin(), input_str_.end(),
                               [](char c) { return std::isalpha(static_cast<unsigned char>(c)); });
  return true;
}

bool muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskSequential::post_processing() {
  reinterpret_cast<int*>(taskData->outputs[0])[0] = alpha_count_;
  return true;
}

bool muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskParallel::pre_processing() {
  internal_order_test();

  int myid = world.rank();
  int world_size = world.size();
  unsigned int n = 0;

  if (myid == 0) {
    n = taskData->inputs_count[0];
    input_str_ = std::vector<char>(n);
    auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    memcpy(input_str_.data(), tmp_ptr, sizeof(char) * n);
  }

  boost::mpi::broadcast(world, n, 0);

  if (world_size == 1) {
    local_input_ = input_str_;
  } else {
    unsigned int vec_send_size = n / world_size;
    unsigned int overflow_size = n % world_size;
    std::vector<int> send_counts(world_size, vec_send_size);
    std::vector<int> displs(world_size, 0);

    for (unsigned int i = 0; i < static_cast<unsigned int>(world_size); ++i) {
      if (i < static_cast<unsigned int>(overflow_size)) {
        ++send_counts[i];
      }
      if (i > 0) {
        displs[i] = displs[i - 1] + send_counts[i - 1];
      }
    }

    unsigned int loc_vec_size = send_counts[myid];
    local_input_.resize(loc_vec_size);

    boost::mpi::scatterv(world, input_str_.data(), send_counts, displs, local_input_.data(), loc_vec_size, 0);
  }

  local_alpha_count_ = 0;
  total_alpha_count_ = 0;
  return true;
}

bool muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskParallel::run() {
  internal_order_test();

  local_alpha_count_ = std::count_if(local_input_.begin(), local_input_.end(),
                                     [](char c) { return std::isalpha(static_cast<unsigned char>(c)); });

  boost::mpi::reduce(world, local_alpha_count_, total_alpha_count_, std::plus<>(), 0);
  std::this_thread::sleep_for(20ms);
  return true;
}

bool muradov_m_count_alpha_chars_mpi::AlphaCharCountTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = total_alpha_count_;
  }

  return true;
}