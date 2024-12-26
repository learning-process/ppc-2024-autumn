// Copyright 2023 Nesterov Alexander
#include "mpi/ermilova_d_Shell_sort_simple_merge/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <thread>
#include <vector>

std::vector<int> ermilova_d_Shell_sort_simple_merge_mpi::ShellSort(std::vector<int>& vec,
                                                                   const std::function<bool(int, int)>& comp) {
  size_t n = vec.size();
  for (size_t gap = n / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < n; i++) {
      int temp = vec[i];
      size_t j;
      for (j = i; j >= gap && comp(vec[j - gap], temp); j -= gap) {
        vec[j] = vec[j - gap];
      }
      vec[j] = temp;
    }
  }
  return vec;
}

std::vector<int> ermilova_d_Shell_sort_simple_merge_mpi::merge(std::vector<int>& vec1, std::vector<int>& vec2,
                                                               const std::function<bool(int, int)>& comp) {
  std::vector<int> result;
  std::merge(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), std::back_inserter(result), comp);

  return result;
}

bool ermilova_d_Shell_sort_simple_merge_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  is_descending = *reinterpret_cast<bool*>(taskData->inputs[1]);
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
  if (is_descending) {
    res = ShellSort(input_, std::less());
  } else {
    res = ShellSort(input_, std::greater());
  }

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
    is_descending = *reinterpret_cast<bool*>(taskData->inputs[1]);
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
  broadcast(world, is_descending, 0);

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
  if (is_descending) {
    local_input_ = ShellSort(local_input_, std::less());
  } else {
    local_input_ = ShellSort(local_input_, std::greater());
  }

  std::vector<std::vector<int>> sorted_inputs;

  gather(world, local_input_, sorted_inputs, 0);

  if (world.rank() == 0) {
    std::vector<int> merge_vec;
    if (is_descending) {
      for (int i = 0; i < world.size(); i++) {
        merge_vec = ermilova_d_Shell_sort_simple_merge_mpi::merge(merge_vec, sorted_inputs[i], std::greater());
      }
    } else {
      for (int i = 0; i < world.size(); i++) {
        merge_vec = ermilova_d_Shell_sort_simple_merge_mpi::merge(merge_vec, sorted_inputs[i], std::less());
      }
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
