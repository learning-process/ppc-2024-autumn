// Copyright 2023 Nesterov Alexander
#include "mpi/ermilova_d_Shell_sort_simple_merge/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <queue>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> ermilova_d_Shell_sort_simple_merge_mpi::ShellSort(std::vector<int>& vec) {
  size_t n = vec.size();
  for (size_t gap = n / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < n; i++) {
      int temp = vec[i];
      size_t j;
      for (j = i; j >= gap && vec[j - gap] > temp; j -= gap) {
        vec[j] = vec[j - gap];
      }
      vec[j] = temp;
    }
  }
  return vec;
}
std::vector<int> ermilova_d_Shell_sort_simple_merge_mpi::merge(std::vector<int>& vec1, std::vector<int>& vec2) {
  std::vector<int> result;
  std::merge(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), std::back_inserter(result));

  return result;
}

bool ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* data = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(data, data + taskData->inputs_count[0], input_.begin());
  return true;
}

bool ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == taskData->inputs_count[0] && taskData->inputs_count[0] > 0;
}

bool ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res = ShellSort(input_);

  return true;
}

bool ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res.begin(), res.end(), data);
  return true;
}

bool ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_.resize(taskData->inputs_count[0]);
    auto* data = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(data, data + taskData->inputs_count[0], input_.begin());
    res.resize(taskData->inputs_count[0]);
  }
  return true;
}

bool ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == taskData->inputs_count[0] && taskData->inputs_count[0] > 0;
  }
  return true;
}

bool ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int rank = world.rank();
  size_t delta = 0;
  size_t extra = 0;
  int size = input_.size();
  std::vector<int> sizes_vec;

  if (world.rank() == 0) {
    delta = input_.size() / world.size();
    extra = input_.size() % world.size();
    sizes_vec = std::vector<int>(world.size(), delta);
    for (int i = 0; i < extra; i++) {
      sizes_vec[i] += 1;
    }
  }
  broadcast(world, size, 0);
  local_input_ = std::vector<int>(delta + (rank < extra ? 1 : 0));

  local_input_ = ShellSort(local_input_);

  scatterv(world, input_, sizes_vec, local_input_.data(), 0);

  std::vector<std::vector<int>> sorted_inputs;

  gather(world, local_input_, sorted_inputs, 0);

  if (world.rank() == 0) {
    std::vector<int> merge_vec;
    for (int i = 0; i < world.size(); i++) {
      merge_vec = ermilova_d_Shell_sort_simple_merge_mpi::merge(merge_vec, sorted_inputs[i]);
    }
    res = merge_vec;
  }

  return true;
}

bool ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res.begin(), res.end(), data);
  }
  return true;
}
