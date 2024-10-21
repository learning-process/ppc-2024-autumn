#include "mpi/shvedova_v_char_freq/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool shvedova_v_char_freq_mpi::CharFrequencySequential::pre_processing() {
  internal_order_test();

  input_str_ = std::vector<char>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_str_[i] = tmp_ptr[i];
  }

  target_char_ = *reinterpret_cast<char*>(taskData->inputs[1]);
  res = 0;
  return true;
}

bool shvedova_v_char_freq_mpi::CharFrequencySequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool shvedova_v_char_freq_mpi::CharFrequencySequential::run() {
  internal_order_test();

  res = std::count(input_str_.begin(), input_str_.end(), target_char_);
  return true;
}

bool shvedova_v_char_freq_mpi::CharFrequencySequential::post_processing() {
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool shvedova_v_char_freq_mpi::CharFrequencyParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;

  if (world.rank() == 0) {
    input_str_ = std::vector<char>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_str_[i] = tmp_ptr[i];
    }

    target_char_ = *reinterpret_cast<char*>(taskData->inputs[1]);

    delta = taskData->inputs_count[0] / world.size();
  }

  boost::mpi::broadcast(world, delta, 0);
  boost::mpi::broadcast(world, target_char_, 0);

  local_input_.resize(delta);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_str_.data() + proc * delta, delta);
    }
    local_input_ = std::vector<char>(input_str_.begin(), input_str_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }

  local_res = 0;
  res = 0;
  return true;
}

bool shvedova_v_char_freq_mpi::CharFrequencyParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool shvedova_v_char_freq_mpi::CharFrequencyParallel::run() {
  internal_order_test();
  local_res = std::count(local_input_.begin(), local_input_.end(), target_char_);

  boost::mpi::reduce(world, local_res, res, std::plus<>(), 0);
  std::this_thread::sleep_for(20ms);
  return true;
}

bool shvedova_v_char_freq_mpi::CharFrequencyParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }

  return true;
}