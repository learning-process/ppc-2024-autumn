// Copyright 2023 Nesterov Alexander
#include "mpi/example/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init vectors
  // Init vector for output
  count_rows = (int)*taskData->inputs[1];
  size_rows = (int)(taskData->inputs_count[0] / count_rows);
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }
  res = std::vector<int>(count_rows, 0);
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check input and count elements of output
  return *taskData->inputs[1] == taskData->outputs_count[0];
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  for (int i = 0; i < count_rows; i++) {
    res[i] = *std::min_element(input_.begin() + i * size_rows, input_.begin() + (i + 1) * size_rows);
  }
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < count_rows; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  count_rows = 0;
  size_rows = 0;

  unsigned int delta_rows = 0;
  unsigned int delta = 0;
  unsigned int count_rank_add = 0;
  unsigned int count_rank = 0;
  if (world.rank() == 0) {
    count_rows = (int)*taskData->inputs[1];
    size_rows = (int)(taskData->inputs_count[0] / count_rows);
    delta_rows = taskData->inputs_count[0] / count_rows / world.size();
    count_rank_add = (taskData->inputs_count[0] / count_rows) % world.size();
    count_rank = (unsigned)world.size() - count_rank_add;
    delta = taskData->inputs_count[0] / count_rows / world.size() * count_rows;
  }
  broadcast(world, count_rows, 0);
  broadcast(world, delta, 0);
  broadcast(world, count_rank, 0);
  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    int ind = 0;
    for (unsigned rank = 0; rank < count_rank; rank++) {
      for (unsigned rows = 0; rows < (unsigned)count_rows; rows++) {
        for (unsigned i = 0; i < delta_rows; i++) {
          input_[ind] = tmp_ptr[i + rows * size_rows + rank * delta_rows];
          ind++;
        }
      }
    }
    for (unsigned rank = 0; rank < count_rank_add; rank++) {
      for (unsigned rows = 0; rows < (unsigned)count_rows; rows++) {
        for (unsigned i = 0; i < delta_rows + 1; i++) {
          input_[ind] = tmp_ptr[i + rows * size_rows + count_rank * delta_rows + rank * (delta_rows + 1)];
          ind++;
        }
      }
    }
    if (delta > 0) {
      for (int proc = 1; proc < (int)count_rank; proc++) {
        world.send(proc, 0, input_.data() + proc * delta, delta);
      }
    }
    for (int proc = count_rank; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + count_rank * delta + (proc - count_rank) * (delta + count_rows),
                 delta + count_rows);
    }
  }
  if (world.rank() == 0) {
    if (delta > 0) {
      local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
    } else {
      local_input_ = std::vector<int>(0);
    }
  } else {
    if (world.rank() < (int)count_rank) {
      if (delta > 0) {
        local_input_ = std::vector<int>(delta);
        world.recv(0, 0, local_input_.data(), delta);
      } else {
        local_input_ = std::vector<int>(0);
      }
    } else {
      local_input_ = std::vector<int>(delta + count_rows);
      world.recv(0, 0, local_input_.data(), delta + count_rows);
    }
  }
  // Init value for output
  res = std::vector<int>(count_rows, 0);
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check input and count elements of output
    return *taskData->inputs[1] == taskData->outputs_count[0];
  }
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int delta = local_input_.size() / count_rows;
  if (delta > 0) {
    for (int i = 0; i < count_rows; i++) {
      int local_res;
      local_res = *std::min_element(local_input_.begin() + i * delta, local_input_.begin() + (i + 1) * delta);
      reduce(world, local_res, res[i], boost::mpi::minimum<int>(), 0);
    }
  } else {
    for (int i = 0; i < count_rows; i++) {
      reduce(world, INT_MAX, res[i], boost::mpi::minimum<int>(), 0);
    }
  }
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < count_rows; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = res[i];
    }
  }
  return true;
}
