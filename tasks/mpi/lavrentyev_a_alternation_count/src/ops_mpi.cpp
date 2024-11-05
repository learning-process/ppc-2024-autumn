#include "mpi/lavrentyev_a_alternation_count/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool lavrentyev_a_alternation_count_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  auto input_size = taskData->inputs_count[0];
  input_ = std::vector<int>(input_ptr, input_ptr + input_size);
  res = 0;
  return true;
}

bool lavrentyev_a_alternation_count_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

bool lavrentyev_a_alternation_count_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (unsigned long i = 0; i < input_.size() - 1; i++) {
    if (((input_[i] * input_[i + 1]) <= 0) && (input_[i] != 0 || input_[i + 1] != 0)) {
      res++;
    }
  }
  return true;
}

bool lavrentyev_a_alternation_count_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}

bool lavrentyev_a_alternation_count_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto input_size = taskData->inputs_count[0];
    auto delta = input_size / world.size();
    auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_ = std::vector<int>(input_ptr, input_ptr + input_size);
    local_vec_ = std::vector<int>(input_ptr, input_ptr + delta + uint32_t(world.size() > 1));
  }
  res = 0;
  return true;
}

bool lavrentyev_a_alternation_count_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool lavrentyev_a_alternation_count_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  if (world.rank() == 0) {
    auto input_size = taskData->inputs_count[0];
    auto delta = input_size / world.size();
    auto* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (int proc = 1; proc < world.size(); proc++) {
      auto start = proc * delta;
      auto size = (proc == world.size() - 1) ? input_size - start : delta + 1;
      world.send(proc, 0, std::vector<int>(input_ptr + start, input_ptr + start + size));
    }
  } else {
    world.recv(0, 0, local_vec_);
  }
  auto localAlternationCount = 0;
  auto vec_size = local_vec_.size();
  for (size_t i = 0; i < vec_size - 1; i++) {
    if (((local_vec_[i] * local_vec_[i + 1]) <= 0) && (local_vec_[i] != 0 || local_vec_[i + 1] != 0)) {
      localAlternationCount++;
    }
  }
  reduce(world, localAlternationCount, res, std::plus(), 0);
  return true;
}

bool lavrentyev_a_alternation_count_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}