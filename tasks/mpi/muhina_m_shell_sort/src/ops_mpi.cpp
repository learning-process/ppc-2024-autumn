// Copyright 2023 Nesterov Alexander
#include "mpi/muhina_m_shell_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

std::vector<int> muhina_m_shell_sort_mpi::shellSort(const std::vector<int>& vect) {
  std::vector<int> sortedVec = vect;
  int n = sortedVec.size();
  int gap;
  for (gap = 1; gap < n / 3; gap = gap * 3 + 1);
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
  int sizeVec = taskData->inputs_count[0];
  int sizeResultVec = taskData->outputs_count[0];

  return (sizeVec > 0 && sizeVec == sizeResultVec);
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
    int sizeVec = taskData->inputs_count[0];
    int sizeResultVec = taskData->outputs_count[0];

    return (sizeVec > 0 && sizeVec == sizeResultVec);
  }
  return true;
}

std::vector<int> muhina_m_shell_sort_mpi::merge(const std::vector<int>& left, const std::vector<int>& right) {
  std::vector<int> result;
  result.reserve(left.size() + right.size());
  std::merge(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(result));

  return result;
}

bool muhina_m_shell_sort_mpi::ShellSortMPIParallel::run() {
  internal_order_test();

  int rank = world_.rank();
  int size = world_.size();

  int n = input_.size();
  broadcast(world_, n, 0);

  int delta = n / size;
  int remainder = n % size;

  std::vector<int> sizes(world_.size(), delta);
  for (int i = 0; i < remainder; ++i) {
    sizes[i]++;
  }

  local_input_.resize(delta + (rank < remainder ? 1 : 0));
  scatterv(world_, input_, sizes, local_input_.data(), 0);

  local_res_ = shellSort(local_input_);

  if (world_.rank() == 0) {
    res_ = local_res_;
    std::vector<int> temp;

    for (int i = 1; i < world_.size(); ++i) {
      world_.recv(i, 0, temp);
      res_ = merge(res_, temp);
    }
  } else {
    world_.send(0, 0, local_res_);
  }

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
