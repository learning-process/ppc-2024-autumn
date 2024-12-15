// Copyright 2023 Nesterov Alexander
#include "mpi/ermilova_d_Shell_sort_simple_merge/include/ops_mpi.hpp"

#include <algorithm>
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
  size_t size = input_.size();
  size_t delta = size / world.size();
  size_t extra = size % world.size();

  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + delta * proc + extra, delta);
    }
  }

  std::vector<int> local_input(delta);

  if (world.rank() == 0) {
    local_input = std::vector<int>(input_.begin(), input_.begin() + delta + extra);
  } else {
    world.recv(0, 0, local_input.data(), delta);
  }

  local_input = ShellSort(local_input);

  std::vector<int> sorted_input;
  if (world.rank() == 0) {
    sorted_input.resize(size);
  }

  boost::mpi::gather(world, local_input.data(), local_input.size(), sorted_input.data(), 0);

  if (world.rank() == 0) {
    std::vector<int> temp_vec = sorted_input;
    sorted_input.clear();
    sorted_input.reserve(size);

    std::priority_queue<std::pair<int, size_t>, std::vector<std::pair<int, size_t>>, std::greater<>> pq;

    for (size_t i = 0; i < world.size(); i++) {
      if (!temp_vec.empty()) {
        pq.push({temp_vec[i * delta], i});
      }
    }

    while (!pq.empty()) {
      auto top = pq.top();
      pq.pop();
      sorted_input.push_back(top.first);

      size_t next_index = top.second * delta + (sorted_input.size() - 1) % delta + 1;
      if (next_index < temp_vec.size() && next_index < (top.second + 1) * delta) {
        pq.push({temp_vec[next_index], top.second});
      }
    }
  }

  res = sorted_input;

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
