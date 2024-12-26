// Copyright 2024 Sdobnov Vladimir
#include "mpi/Sdobnov_V_mergesort_Betcher/include/ops_mpi.hpp"

#include <random>
#include <vector>

std::vector<int> Sdobnov_V_mergesort_Betcher_par::generate_random_vector(int size, int lower_bound, int upper_bound) {
  std::vector<int> res(size);
  for (int i = 0; i < size; i++) {
    res[i] = lower_bound + rand() % (upper_bound - lower_bound + 1);
  }
  return res;
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

  std::sort(local_vec_.begin(), local_vec_.end());

  for (int step = 0; step < size; step++) {
    if (rank % 2 == 0) {
      if (step % 2 == 0) {
        if (rank + 1 < size) {
          for (int i = 0; i < counts[rank + 1]; i++) {
            int tmp;
            world.recv(rank + 1, 0, tmp);
            local_vec_.push_back(tmp);
          }
          std::sort(local_vec_.begin(), local_vec_.end());
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
          std::sort(local_vec_.begin(), local_vec_.end());
          for (int i = local_vec_.size() - 1; i >= counts[rank]; i--) {
            world.send(rank + 1, 0, local_vec_[i]);
            local_vec_.pop_back();
          }
        }
      }
    }
    std::sort(local_vec_.begin(), local_vec_.end());
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