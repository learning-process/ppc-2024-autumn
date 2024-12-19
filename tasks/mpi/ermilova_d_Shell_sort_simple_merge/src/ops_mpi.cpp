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

  std::vector<size_t> Sedgwick_gaps;

  size_t a = 0;
  while (true) {
    size_t gap;
    if (a % 2 == 0) {
      gap = 9 * static_cast<size_t>(pow(2, a)) - 9 * static_cast<size_t>(pow(2, a / 2)) + 1;
    } else {
      gap = 8 * static_cast<size_t>(pow(2, a)) - 6 * static_cast<size_t>(pow(2, (a + 1) / 2)) + 1;
    }

    if (gap >= n) break;
    Sedgwick_gaps.push_back(gap);
    a++;
  }
  for (int k = Sedgwick_gaps.size() - 1; k >= 0; --k) {
    size_t gap = Sedgwick_gaps[k];

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
  int delta = 0;
  int extra = 0;
  int size = input_.size();
  broadcast(world, size, 0);

  std::vector<int> sizes_vec;

  delta = size / world.size();
  extra = size % world.size();

  if (world.rank() == 0) {
    sizes_vec.resize(world.size(), delta);
    for (int i = 0; i < extra; i++) {
      sizes_vec[i] += 1;
    }
  }
  local_input_.resize(delta + (rank < extra ? 1 : 0));
  if (world.rank() == 0) {
    scatterv(world, input_, sizes_vec, local_input_.data(), 0);
  } else {
    scatterv(world, local_input_.data(), local_input_.size(), 0);
  }

  local_input_ = ShellSort(local_input_);

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
