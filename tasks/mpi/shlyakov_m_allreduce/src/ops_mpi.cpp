// Copyright 2023 Nesterov Alexander
#include "mpi/shlyakov_m_allreduce/include/ops_mpi.hpp"

template <typename T>
void shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel::my_all_reduce(const boost::mpi::communicator& comm,
                                                                     const T& value, T& out_value) {
  unsigned int rank = comm.rank();
  unsigned int size = comm.size();
  unsigned int id_child_1 = 2 * rank + 1;
  unsigned int id_child_2 = 2 * rank + 2;
  unsigned int id_parent = (rank - 1) >> 1;

  out_value = value;

  T child_1_value;
  T child_2_value;
  bool received_child_1 = false;
  bool received_child_2 = false;

  if (id_child_1 < size) {
    comm.recv(id_child_1, 0, child_1_value);
    received_child_1 = true;
  }
  if (id_child_2 < size) {
    comm.recv(id_child_2, 0, child_2_value);
    received_child_2 = true;
  }

  if (received_child_1) {
    out_value = std::min(out_value, child_1_value);
  }
  if (received_child_2) {
    out_value = std::min(out_value, child_2_value);
  }

  if (rank != 0) {
    comm.send(id_parent, 0, out_value);
    comm.recv(id_parent, 0, out_value);
  }

  if (id_child_1 < size) {
    comm.send(id_child_1, 0, out_value);
  }
  if (id_child_2 < size) {
    comm.send(id_child_2, 0, out_value);
  }
}

std::vector<int> shlyakov_m_all_reduce_mpi::TestMPITaskSequential::generate_matrix(int row, int col) {
  std::vector<int> tmp(row * col);

  std::random_device rd;
  std::mt19937 generate_matrix(rd());
  std::uniform_int_distribution<int> dist(0, col - 1);

  for (int i = 0; i < row; i++) {
    int col_index = dist(generate_matrix);
    tmp[col_index + i * col] = INT_MIN;
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
    input_ = std::vector<int>(delta * world.size(), INT_MAX);
    std::copy(reinterpret_cast<int*>(taskData->inputs[0]), reinterpret_cast<int*>(taskData->inputs[0]) + size,
              input_.begin());
    res.resize(row, 0);
  }

  broadcast(world, row, 0);
  broadcast(world, col, 0);
  broadcast(world, size, 0);
  broadcast(world, delta, 0);
  broadcast(world, res_, 0);

  local_input_.resize(delta);
  boost::mpi::scatter(world, input_.data(), local_input_.data(), delta, 0);

  int l_res = *std::min_element(local_input_.begin(), local_input_.begin() + delta);
  MyTestMPITaskParallel::my_all_reduce(world, l_res, res_);

  std::vector<int> ress(row, 0);
  int count = 0;
  for (int id = 0; id < delta; ++id) {
    int global_id = id + delta * world.rank();
    if ((global_id % col == 0) && (world.rank() != 0 || id != 0)) {
      ress[(global_id / col) - static_cast<int>(global_id % col == 0)] += count;
      count = 0;
    }
    if (global_id >= size) break;
    if (local_input_[id] == res_) {
      count++;
    }
  }

  if (count > 0) {
    int row_index = (world.rank() * delta + delta - 1) / col;
    ress[row_index] += count;
  }

  boost::mpi::reduce(world, ress.data(), row, res.data(), std::plus(), 0);

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
    size = col * row;
    delta = (size + world.size() - 1) / world.size();
    input_ = std::vector<int>(delta * world.size(), INT_MAX);
    std::copy(reinterpret_cast<int*>(taskData->inputs[0]), reinterpret_cast<int*>(taskData->inputs[0]) + size,
              input_.begin());
    res.resize(row, 0);
  }

  broadcast(world, row, 0);
  broadcast(world, col, 0);
  broadcast(world, size, 0);
  broadcast(world, delta, 0);
  broadcast(world, res_, 0);

  local_input_.resize(delta);
  boost::mpi::scatter(world, input_.data(), local_input_.data(), delta, 0);

  int l_res = *std::min_element(local_input_.begin(), local_input_.begin() + delta);
  boost::mpi::all_reduce(world, l_res, res_, boost::mpi::minimum<int>());

  std::vector<int> ress(row, 0);
  int count = 0;
  for (int id = 0; id < delta; ++id) {
    int global_id = id + delta * world.rank();
    if ((global_id % col == 0) && (world.rank() != 0 || id != 0)) {
      ress[(global_id / col) - static_cast<int>(global_id % col == 0)] += count;
      count = 0;
    }
    if (global_id >= size) break;
    if (local_input_[id] == res_) {
      count++;
    }
  }

  if (count > 0) {
    int row_index = (world.rank() * delta + delta - 1) / col;
    ress[row_index] += count;
  }

  boost::mpi::reduce(world, ress.data(), row, res.data(), std::plus(), 0);

  return true;
}

bool shlyakov_m_all_reduce_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    std::copy(res.begin(), res.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }

  return true;
}