// Copyright 2023 Nesterov Alexander
#include "mpi/sorokin_a_check_lexicographic_order_of_strings/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<std::vector<char>>(taskData->inputs_count[0], std::vector<char>(taskData->inputs_count[1]));

  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i][j] = tmp_ptr[j];
    }
  }
  res_ = 0;
  return true;
}

bool sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 2 && taskData->outputs_count[0] == 1;
}

bool sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < std::min(input_[0].size(), input_[1].size()); ++i) {
    if (static_cast<int>(input_[0][i]) > static_cast<int>(input_[1][i])) {
      res_ = 1;
      break;
    }
    if (static_cast<int>(input_[0][i]) < static_cast<int>(input_[1][i])) {
      break;
    }
  }
  return true;
}

bool sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}

bool sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[1] / world.size();
  }
  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    input_ = std::vector<std::vector<char>>(taskData->inputs_count[0], std::vector<char>(taskData->inputs_count[1]));

    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[i]);
      for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
        input_[i][j] = tmp_ptr[j];
      }
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_[0].data() + delta * proc, delta);
      world.send(proc, 1, input_[1].data() + delta * proc, delta);
    }
  }
  local_input1_ = std::vector<char>(delta);
  local_input2_ = std::vector<char>(delta);
  if (world.rank() == 0) {
    local_input1_ = std::vector<char>(input_[0].begin(), input_[0].begin() + delta);
    local_input2_ = std::vector<char>(input_[1].begin(), input_[1].begin() + delta);
  } else {
    world.recv(0, 0, local_input1_.data(), delta);
    world.recv(0, 1, local_input2_.data(), delta);
  }
  res_ = 2;
  return true;
}

bool sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int local_res = 2;
  for (size_t i = 0; i < local_input1_.size(); ++i) {
    if (static_cast<int>(local_input1_[i]) > static_cast<int>(local_input2_[i])) {
      local_res = 1;
      break;
    }
    if (static_cast<int>(local_input1_[i]) < static_cast<int>(local_input2_[i])) {
      local_res = 0;
      break;
    }
  }
  std::vector<int> all_res;
  boost::mpi::gather(world, local_res, all_res, 0);

  if (world.rank() == 0) {
    for (int result : all_res) {
      if (result != 2) {
        res_ = result;
        break;
      }
    }
  }
  return true;
}

bool sorokin_a_check_lexicographic_order_of_strings_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}
