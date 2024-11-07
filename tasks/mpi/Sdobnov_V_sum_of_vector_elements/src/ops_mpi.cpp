// Copyright 2024 Sdobnov Vladimir
#include "mpi/Sdobnov_V_sum_of_vector_elements/include/ops_mpi.hpp"

#include <random>
#include <vector>

int Sdobnov_V_sum_of_vector_elements::vec_elem_sum(const std::vector<int>& vec) {
  int res = 0;
  for (int elem : vec) {
    res += elem;
  }
  return res;
}

bool Sdobnov_V_sum_of_vector_elements::SumVecElemSequential::pre_processing() {
  internal_order_test();

  int rows = taskData->inputs_count[0];
  int columns = taskData->inputs_count[1];

  input_ = std::vector<int>(rows * columns);

  for (int i = 0; i < rows; i++) {
    auto* p = reinterpret_cast<int*>(taskData->inputs[i]);
    for (int j = 0; j < columns; j++) {
      input_[i * columns + j] = p[j];
    }
  }

  return true;
}

bool Sdobnov_V_sum_of_vector_elements::SumVecElemSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count.size() == 2 && taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 &&
          taskData->outputs_count[0] == 1);
}

bool Sdobnov_V_sum_of_vector_elements::SumVecElemSequential::run() {
  internal_order_test();
  res_ = vec_elem_sum(input_);
  return true;
}

bool Sdobnov_V_sum_of_vector_elements::SumVecElemSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}

bool Sdobnov_V_sum_of_vector_elements::SumVecElemParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int rows = taskData->inputs_count[0];
    int columns = taskData->inputs_count[1];

    input_ = std::vector<int>(rows * columns);

    for (int i = 0; i < rows; i++) {
      auto* p = reinterpret_cast<int*>(taskData->inputs[i]);
      for (int j = 0; j < columns; j++) {
        input_[i * columns + j] = p[j];
      }
    }
  }

  return true;
}

bool Sdobnov_V_sum_of_vector_elements::SumVecElemParallel::validation() {
  internal_order_test();
  if (world.rank() == 0)
    return (taskData->inputs_count.size() == 2 && taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 &&
            taskData->outputs_count.size() == 1 && taskData->outputs_count[0] == 1);
  return true;
}

bool Sdobnov_V_sum_of_vector_elements::SumVecElemParallel::run() {
  internal_order_test();

  int input_size = 0;
  int local_rank = world.rank();
  int world_size = world.size();
  if (local_rank == 0) input_size = input_.size();
  boost::mpi::broadcast(world, input_size, 0);

  int elem_per_procces = input_size / world_size;
  int residual_elements = input_size % world_size;

  int process_count = elem_per_procces + (local_rank < residual_elements ? 1 : 0);

  std::vector<int> counts(world_size);
  std::vector<int> displacment(world_size);

  for (int i = 0; i < world_size; i++) {
    counts[i] = elem_per_procces + (i < residual_elements ? 1 : 0);
    displacment[i] = i * elem_per_procces + std::min(i, residual_elements);
  }

  local_input_.resize(counts[local_rank]);
  boost::mpi::scatterv(world, input_.data(), counts, displacment, local_input_.data(), process_count, 0);

  int process_sum = vec_elem_sum(local_input_);
  boost::mpi::reduce(world, process_sum, res_, std::plus(), 0);

  return true;
}

bool Sdobnov_V_sum_of_vector_elements::SumVecElemParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}