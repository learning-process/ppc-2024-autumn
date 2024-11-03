// Copyright 2023 Nesterov Alexander
#include "mpi/beresnev_a_min_values_by_matrix_columns/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  input_ = reinterpret_cast<std::vector<int>*>(taskData->inputs[0])[0];
  res_ = reinterpret_cast<std::vector<int>*>(taskData->outputs[0])[0];
  n_ = reinterpret_cast<int*>(taskData->inputs[1])[0];
  m_ = reinterpret_cast<int*>(taskData->inputs[2])[0];
  return true;
}

bool beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] > 0 &&
         taskData->inputs_count[0] == reinterpret_cast<std::vector<int>*>(taskData->inputs[0])[0].size() &&
         taskData->inputs_count[0] == (uint32_t) reinterpret_cast<int*>(taskData->inputs[1])[0] *
                                          reinterpret_cast<int*>(taskData->inputs[2])[0] &&
         taskData->outputs_count[0] == reinterpret_cast<std::vector<int>*>(taskData->outputs[0])[0].size();
}

bool beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < m_; i++) {
    int min = input_[i];
    for (int j = 1; j < n_; j++) {
      if (input_[j * m_ + i] < min) {
        min = input_[j * m_ + i];
      }
    }
    res_[i] = min;
  }
  return true;
}

bool beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<int>*>(taskData->outputs[0])[0] = res_;
  return true;
}

bool beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    n_ = reinterpret_cast<int*>(taskData->inputs[1])[0];
    m_ = reinterpret_cast<int*>(taskData->inputs[2])[0];
    col_on_pr = m_ / world.size();
    remainder = m_ % world.size();
    int total_elements = taskData->inputs_count[0];
    input_.resize(total_elements);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (int i = 0; i < total_elements; i++) {
      input_[i] = tmp_ptr[i];
    }

    for (int proc = 1; proc < world.size(); proc++) {
      req = world.isend(proc, 0, input_.data() + (col_on_pr * proc + remainder) * n_, n_ * col_on_pr);
    }
  }

  return true;
}

bool beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0 &&
           taskData->inputs_count[0] == (uint32_t) reinterpret_cast<int*>(taskData->inputs[1])[0] *
                                            reinterpret_cast<int*>(taskData->inputs[2])[0] &&
           taskData->outputs_count[0] == (uint32_t) reinterpret_cast<int*>(taskData->inputs[2])[0] &&
           taskData->inputs_count[1] == 1 && taskData->inputs_count[2] == 1;
  }
  return true;
}

bool beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  broadcast(world, col_on_pr, 0);
  broadcast(world, remainder, 0);
  broadcast(world, n_, 0);
  broadcast(world, m_, 0);

  local_input_ = std::vector<int>(n_ * col_on_pr);
  local_mins_ = std::vector<int>(col_on_pr);
  global_mins_ = std::vector<int>(m_, 0);
  if (world.rank() == 0) {
    req.wait();
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + (col_on_pr + remainder) * n_);
    local_mins_.resize(col_on_pr + remainder);
  } else {
    world.recv(0, 0, local_input_.data(), local_input_.size());
  }

  if (world.rank() == 0) {
    for (int i = 0; i < col_on_pr + remainder; i++) {
      local_mins_[i] = *std::min_element(local_input_.begin() + n_ * i, local_input_.begin() + n_ * (i + 1));
    }
  } else {
    for (int i = 0; i < col_on_pr; i++) {
      local_mins_[i] = *std::min_element(local_input_.begin() + n_ * i, local_input_.begin() + n_ * (i + 1));
    }
  }

  std::vector<int> sizes(world.size(), col_on_pr);
  sizes[0] += remainder;

  boost::mpi::gatherv(world, local_mins_, global_mins_.data(), sizes, 0);

  return true;
}

bool beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    for (size_t j = 0; j < global_mins_.size(); j++) {
      output_ptr[j] = global_mins_[j];
    }
  }
  return true;
}