// Copyright 2024 Nesterov Alexander
#include "mpi/sidorina_p_check_lexicographic_order/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_.resize(taskData->inputs_count[0]);
  for (unsigned int i = 0; i < 2; i++) input_[i].resize(taskData->inputs_count[i + 1]);
  for (size_t i = 0; i < taskData->inputs_count[0]; ++i) {
    const char* tmp_ptr = reinterpret_cast<const char*>(taskData->inputs[i]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], input_[i].begin());
  }
  res = 0;
  return true;
}

bool sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 2 && taskData->outputs_count[0] == 1;
}

bool sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (size_t i = 0; i < std::min(input_[0].size(), input_[1].size()); ++i) {
    if (input_[0][i] == input_[1][i])
      res = 2;
    if (input_[0][i] > input_[1][i]) {
      res = 1;
      break;
    }
    if (input_[0][i] < input_[1][i]) {
      res = 0;
      break;
    }
  }
  if (res == 2 && input_[0].size() != input_[1].size()) {
    if (input_[0].size() > input_[1].size()) {
      res = 1;
    } else {
      res = 0;
    }
  }
  return true;
}

bool sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = std::min(taskData->inputs_count[1], taskData->inputs_count[2]) / world.size();
  }
  broadcast(world, delta, 0);
  if (world.rank() == 0) {
    input_.resize(taskData->inputs_count[0]);
    for (unsigned int i = 0; i < 2; i++) input_[i].resize(taskData->inputs_count[i + 1]);
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[i]);
      std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], input_[i].begin());
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_[0].data() + delta * proc, delta);
      world.send(proc, 1, input_[1].data() + delta * proc, delta);
    }
  }
  if (world.rank() == 0) {
    local_input1_ = std::vector<char>(input_[0].begin(), input_[0].begin() + delta);
    local_input2_ = std::vector<char>(input_[1].begin(), input_[1].begin() + delta);
  } else {
    local_input1_.resize(delta);
    local_input2_.resize(delta);
    world.recv(0, 0, local_input1_.data(), delta);
    world.recv(0, 1, local_input2_.data(), delta);
  }
  res = 2;
  return true;
}

bool sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int local_res = 2;
  for (size_t i = 0; i < local_input1_.size(); i++) {
    if (local_input1_[i] > local_input2_[i]) {
      local_res = 1;
      break;
    }
    if (local_input1_[i] < local_input2_[i]) {
      local_res = 0;
      break;
    }
  }
  std::vector<int> full_result;
  boost::mpi::gather(world, local_res, full_result, 0);
  if (world.rank() == 0) {
    for (int result : full_result) {
      if (result != 2) {
        res = result;
        break;
      }
    }
    if (res == 2 && input_[0].size() != input_[1].size()) {
      if (input_[0].size() > input_[1].size()) {
        res = 1;
      } else {
        res = 0;
      }
    }
  }
  return true;
}
bool sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}