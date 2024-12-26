// Copyright 2024 Sdobnov Vladimir
#include "mpi/Sdobnov_V_mergesort_Betcher/include/ops_mpi.hpp"

#include <random>
#include <stack>
#include <vector>

std::vector<int> Sdobnov_V_mergesort_Betcher_par::generate_random_vector(int size, int lower_bound, int upper_bound) {
  std::vector<int> res(size);
  for (int i = 0; i < size; i++) {
    res[i] = lower_bound + rand() % (upper_bound - lower_bound + 1);
  }
  return res;
}

int Sdobnov_V_mergesort_Betcher_par::partition(std::vector<int>& vec, int low, int high) {
  int pivot = vec[high];
  int i = low - 1;

  for (int j = low; j < high; ++j) {
    if (vec[j] <= pivot) {
      i++;
      std::swap(vec[i], vec[j]);
    }
  }

  std::swap(vec[i + 1], vec[high]);
  return i + 1;
}

void Sdobnov_V_mergesort_Betcher_par::quickSortIterative(std::vector<int>& vec, int low, int high) {
  std::stack<std::pair<int, int>> s;
  s.emplace(low, high);

  while (!s.empty()) {
    auto [l, h] = s.top();
    s.pop();
    if (l < h) {
      int pi = partition(vec, l, h);
      s.emplace(l, pi - 1);
      s.emplace(pi + 1, h);
    }
  }
}

bool Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    size_ = taskData->inputs_count[0];
    input_.assign(size_, 0);

    auto* input = reinterpret_cast<int*>(taskData->inputs[0]);

    std::copy(input, input + size_, input_.begin());
    for (int i = 0; i < size_; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = input_[i];
    }
  }

  return true;
}

bool Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar::validation() {
  internal_order_test();
  if (world.rank() == 0)
    return (taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 0 && taskData->inputs.size() == 1 &&
            taskData->outputs_count.size() == 1 && taskData->outputs_count[0] >= 0 && taskData->outputs.size() == 1);
  return true;
}

bool Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar::run() {
  internal_order_test();

  int input_size = 0;
  int rank = world.rank();
  int size = world.size();
  if (rank == 0) input_size = size_;
  boost::mpi::broadcast(world, input_size, 0);

  int elem_per_procces = input_size / size;
  int residual_elements = input_size % size;

  int process_count = elem_per_procces + (rank < residual_elements ? 1 : 0);

  std::vector<int> counts(size);
  std::vector<int> displacment(size);

  for (int i = 0; i < size; i++) {
    counts[i] = elem_per_procces + (i < residual_elements ? 1 : 0);
    displacment[i] = i * elem_per_procces + std::min(i, residual_elements);
  }

  local_vec_.resize(counts[rank]);
  boost::mpi::scatterv(world, input_.data(), counts, displacment, local_vec_.data(), process_count, 0);

  quickSortIterative(local_vec_, 0, counts[rank] - 1);

  for (int step = 0; step < size; step++) {
    if (rank % 2 == 0) {
      if (step % 2 == 0) {
        if (rank + 1 < size) {
          for (int i = 0; i < counts[rank + 1]; i++) {
            int tmp;
            world.recv(rank + 1, 0, tmp);
            local_vec_.push_back(tmp);
          }
          quickSortIterative(local_vec_, 0, counts[rank] + counts[rank + 1] - 1);
          for (int i = local_vec_.size() - 1; i >= counts[rank]; i--) {
            world.send(rank + 1, 0, local_vec_[i]);
            local_vec_.pop_back();
          }
        }
      } else {
        if (rank - 1 > 0) {
          for (int i = 0; i < counts[rank]; i++) {
            world.send(rank - 1, 0, local_vec_[i]);
          }
          for (int i = local_vec_.size() - 1; i >= 0; i--) {
            world.recv(rank - 1, 0, local_vec_[i]);
          }
        }
      }
    } else {
      if (step % 2 == 0) {
        for (int i = 0; i < counts[rank]; i++) {
          world.send(rank - 1, 0, local_vec_[i]);
        }
        for (int i = local_vec_.size() - 1; i >= 0; i--) {
          world.recv(rank - 1, 0, local_vec_[i]);
        }
      } else {
        if (rank + 1 < size) {
          for (int i = 0; i < counts[rank + 1]; i++) {
            int tmp;
            world.recv(rank + 1, 0, tmp);
            local_vec_.push_back(tmp);
          }
          quickSortIterative(local_vec_, 0, counts[rank] + counts[rank + 1] - 1);
          for (int i = local_vec_.size() - 1; i >= counts[rank]; i--) {
            world.send(rank + 1, 0, local_vec_[i]);
            local_vec_.pop_back();
          }
        }
      }
    }
    quickSortIterative(local_vec_, 0, counts[rank] - 1);
  }
  boost::mpi::gather(world, local_vec_.data(), counts[rank], input_.data(), 0);

  return true;
}

bool Sdobnov_V_mergesort_Betcher_par::MergesortBetcherPar::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < size_; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = input_[i];
    }
  }
  return true;
}