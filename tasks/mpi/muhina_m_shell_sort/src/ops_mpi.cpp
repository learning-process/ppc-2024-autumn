// Copyright 2023 Nesterov Alexander
#include "mpi/muhina_m_shell_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> muhina_m_shell_sort_mpi::shellSort(const std::vector<int>& vect) {
  std::vector<int> sortedVec = vect;
  int n = sortedVec.size();
  int gap;
  for (gap = 1; gap < n / 3; gap = gap * 3 + 1)
    ;
  for (; gap > 0; gap = (gap - 1) / 3) {
    for (int i = gap; i < n; i++) {
      int temp = sortedVec[i];
      int j;
      for (j = i; j >= gap && sortedVec[j - gap] > temp; j -= gap) {
        sortedVec[j] = sortedVec[j - gap];
      }
      sortedVec[j] = temp;
    }
  }
  return sortedVec;
}
bool muhina_m_shell_sort_mpi::ShellSortMPISequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  return true;
}

bool muhina_m_shell_sort_mpi::ShellSortMPISequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] == 0) {
    return taskData->outputs_count[0] == 0;
  }
  return taskData->outputs_count[0] != 0;
}

bool muhina_m_shell_sort_mpi::ShellSortMPISequential::run() {
  internal_order_test();
  res_ = shellSort(input_);
  return true;
}

bool muhina_m_shell_sort_mpi::ShellSortMPISequential::post_processing() {
  internal_order_test();
  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), output_data);
  return true;
}

bool muhina_m_shell_sort_mpi::ShellSortMPIParallel::pre_processing() {
  internal_order_test();

  if (world_.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  }
  return true;
}

bool muhina_m_shell_sort_mpi::ShellSortMPIParallel::validation() {
  internal_order_test();
  if (world_.rank() == 0) {
    if (taskData->inputs_count[0] == 0) {
      return taskData->outputs_count[0] == 0;
    }
    return taskData->outputs_count[0] != 0;
  }
  return true;
}

bool muhina_m_shell_sort_mpi::ShellSortMPIParallel::run() {
  internal_order_test();

  int rank = world_.rank();
  int size = world_.size();
  int n = taskData->inputs_count[0];

  int delta = n / size;
  int remainder = n % size;

  std::vector<int> local_input(delta + (rank < remainder ? 1 : 0));

  if (rank == 0) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
      int count = delta + (i < remainder ? 1 : 0);
      if (i == 0) {
        std::copy(tmp_ptr, tmp_ptr + count, local_input.begin());
      } else {
        world_.send(i, 0, tmp_ptr + offset, count);
      }
      offset += count;
    }
  } else {
    world_.recv(0, 0, local_input.data(), local_input.size());
  }
  world_.barrier();
  local_input = shellSort(local_input);
  world_.barrier();
  if (rank == 0) {
    res_.resize(n);
    std::copy(local_input.begin(), local_input.end(), res_.begin());
    int offset = local_input.size();
    for (int i = 1; i < size; ++i) {
      int count = delta + (i < remainder ? 1 : 0);
      std::vector<int> recv_data(count);
      world_.recv(i, 1, recv_data.data(), count);
      std::copy(recv_data.begin(), recv_data.end(), res_.begin() + offset);
      offset += count;
    }

  } else {
    world_.send(0, 1, local_input.data(), local_input.size());
  }
  world_.barrier();
  return true;
}

bool muhina_m_shell_sort_mpi::ShellSortMPIParallel::post_processing() {
  internal_order_test();
  if (world_.rank() == 0) {
    int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), output_data);
  }
  return true;
}
