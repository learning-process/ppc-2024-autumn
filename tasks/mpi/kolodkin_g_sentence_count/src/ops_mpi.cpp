#include "mpi/kolodkin_g_sentence_count/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool kolodkin_g_sentence_count_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<char>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  res = 0;
  return true;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (unsigned long i = 0; i < input_.size(); i++) {
    if ((input_[i] == '.' || input_[i] == '!' || input_[i] == '?') &&
        ((input_[i + 1] != '.' && input_[i + 1] != '!' && input_[i + 1] != '?') || i + 1 == input_.size())) {
      res++;
    }
  }
  return true;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_ = std::vector<char>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  }
  localSentenceCount = 0;
  res = 0;
  return true;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
  }
  broadcast(world, delta, 0);
  local_input_.resize(delta);
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
    local_input_ = std::vector<char>(input_.begin(), input_.begin() + delta);
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }
  for (unsigned long i = 0; i < local_input_.size(); i++) {
    if ((local_input_[i] == '.' || local_input_[i] == '!' || local_input_[i] == '?') &&
        ((local_input_[i + 1] != '.' && local_input_[i + 1] != '!' && local_input_[i + 1] != '?') ||
         i + 1 == local_input_.size())) {
      localSentenceCount++;
    }
  }
  reduce(world, localSentenceCount, res, std::plus<>(), 0);
  return true;
}

bool kolodkin_g_sentence_count_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
