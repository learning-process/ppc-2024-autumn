// Copyright 2023 Nesterov Alexander
#include "mpi/shlyakov_m_allreduce/include/ops_mpi.hpp"

#include <algorithm>
#include <random>
#include <string>
#include <vector>

template <typename T>
void shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel::my_all_reduce(const boost::mpi::communicator& comm,
                                                                     const T& value, T& out_value) {
  int rank = comm.rank();
  int size = comm.size();

  out_value = value;

  for (int level = 0; (1 << level) < size; ++level) {
    int parent = (rank >> (level + 1)) << (level + 1);  // Parent index
    int left_child = parent + (1 << level);
    int right_child = parent + (1 << level) + 1;

    if (left_child < size) {
      T child_value;
      comm.recv(left_child, 0, &child_value, 1);
      out_value = std::min(out_value, child_value);
    }
    if (right_child < size) {
      T child_value;
      comm.recv(right_child, 0, &child_value, 1);
      out_value = std::min(out_value, child_value);
    }
  }

  for (int level = 0; (1 << level) < size; ++level) {
    int parent = (rank >> (level + 1)) << (level + 1);
    int left_child = parent + (1 << level);
    int right_child = parent + (1 << level) + 1;
    if (left_child < size) {
      comm.send(left_child, 0, &out_value, 1);
    }
    if (right_child < size) {
      comm.send(right_child, 0, &out_value, 1);
    }
  }
}

std::vector<int> shlyakov_m_all_reduce_mpi::TestMPITaskSequential::generate_matrix(int row, int col) {
  std::vector<int> tmp(row * col);
  int min_val = std::min(col, row);
  int max_val = std::max(col, row) + 7;

  std::random_device rd;
  std::mt19937 generate_matrix(rd());
  std::uniform_int_distribution<int> dist(0, col - 1);

  for (int i = 0; i < row; i++) {
    int col_index = dist(generate_matrix);
    tmp[col_index + i * col] = INT_MIN;  // in 1 of col columns the value must be INT_MIN (needed to check answer)
  }

  return tmp;
}

bool shlyakov_m_all_reduce_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  int row = taskData->inputs_count[0];
  int col = taskData->inputs_count[1];
  int size = row * col;
  input_.resize(size);
  res.resize(row);

  std::copy(reinterpret_cast<int*>(taskData->inputs[0]), reinterpret_cast<int*>(taskData->inputs[0]) + size,
            input_.begin());

  return true;
}

bool shlyakov_m_all_reduce_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
          (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
          (taskData->outputs_count[0] == taskData->inputs_count[0]));
}

bool shlyakov_m_all_reduce_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  int row = taskData->inputs_count[0];
  int col = taskData->inputs_count[1];

  int res_ = *std::min_element(input_.begin(), input_.end());

  std::vector<int> counts(row, 0);
  for (int i = 0; i < row * col; i++) {
    if (input_[i] == res_) {
      counts[i / col]++;
    }
  }
  std::copy(counts.begin(), counts.end(), res.begin());

  return true;
}

bool shlyakov_m_all_reduce_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));

  return true;
}

bool shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel::pre_processing() {
  internal_order_test();
  /* int res_ = 0;
  int row = 0;
  int col = 0;
  int size = 0;
  int delta = 0;
  */
  return true;
}

bool shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
            (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
            (taskData->outputs_count[0] == taskData->inputs_count[0]));
  }

  return true;
}

bool shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel::run() {
  internal_order_test();

  int res_ = 0;
  int row = 0;
  int col = 0;
  int size = 0;
  int delta = 0;

  if (world.rank() == 0) {
    res_ = INT_MAX;
    row = taskData->inputs_count[0];
    col = taskData->inputs_count[1];
    size = col * row;
    delta = (size + world.size() - 1) / world.size();
    input_.resize(size, INT_MAX);
    std::copy(reinterpret_cast<int*>(taskData->inputs[0]), reinterpret_cast<int*>(taskData->inputs[0]) + size,
              input_.begin());
    res.resize(row, 0);
  }

  boost::mpi::broadcast(world, row, 0);
  boost::mpi::broadcast(world, col, 0);
  boost::mpi::broadcast(world, delta, 0);
  boost::mpi::broadcast(world, res_, 0);

  local_input_.resize(delta);
  boost::mpi::scatter(world, input_.data(), local_input_.data(), delta, 0);

  int l_res = *std::min_element(local_input_.begin(), local_input_.end());
  MyTestMPITaskParallel::my_all_reduce(world, l_res, res_);

  std::vector<int> ress(row, 0);
  for (int i = 0; i < local_input_.size(); ++i) {
    if (local_input_[i] == res_) {
      ress[i / col]++;
    }
  }

  boost::mpi::reduce(world, ress.data(), ress.size(), res.data(), std::plus<int>(), 0);

  return true;
}

bool shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }

  return true;
}

bool shlyakov_m_all_reduce_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool shlyakov_m_all_reduce_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return ((!taskData->inputs.empty() && !taskData->outputs.empty()) &&
            (taskData->inputs_count.size() >= 2 && taskData->inputs_count[0] != 0 && taskData->inputs_count[1] != 0) &&
            (taskData->outputs_count[0] == taskData->inputs_count[0]));
  }

  return true;
}

bool shlyakov_m_all_reduce_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int res_ = 0;
  int row = 0;
  int col = 0;
  int size = 0;
  int delta = 0;

  if (world.rank() == 0) {
    res_ = INT_MAX;
    row = taskData->inputs_count[0];
    col = taskData->inputs_count[1];
    size = row * col;
    delta = (size + world.size() - 1) / world.size();
    input_.resize(size, INT_MAX);
    std::copy(reinterpret_cast<int*>(taskData->inputs[0]), reinterpret_cast<int*>(taskData->inputs[0]) + size,
              input_.begin());
    res.resize(row, 0);
  }

  boost::mpi::broadcast(world, row, 0);
  boost::mpi::broadcast(world, col, 0);
  boost::mpi::broadcast(world, delta, 0);

  local_input_.resize(delta);
  boost::mpi::scatter(world, input_.data(), local_input_.data(), delta, 0);

  int l_res = local_input_.empty() ? INT_MAX : *std::min_element(local_input_.begin(), local_input_.end());
  boost::mpi::all_reduce(world, l_res, res_, boost::mpi::minimum<int>());

  std::vector<int> ress(row, 0);
  for (size_t i = 0; i < local_input_.size(); ++i) {
    if (local_input_[i] == res_) {
      ress[i / col]++;
    }
  }

  boost::mpi::reduce(world, ress.data(), ress.size(), res.data(), std::plus<int>(), 0);

  return true;
}

bool shlyakov_m_all_reduce_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }

  return true;
}