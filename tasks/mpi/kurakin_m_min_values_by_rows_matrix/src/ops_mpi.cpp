// Copyright 2023 Nesterov Alexander
#include "mpi/kurakin_m_min_values_by_rows_matrix/include/ops_mpi.hpp"

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

  count_rows = (int)*taskData->inputs[1];
  size_rows = (int)*taskData->inputs[2];
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

  return taskData->inputs.size() == 3 && taskData->inputs_count.size() == 3 && taskData->outputs.size() == 1 &&
         taskData->outputs_count.size() == 1 && *taskData->inputs[1] != 0 && *taskData->inputs[2] != 0 &&
         *taskData->inputs[1] == taskData->outputs_count[0];
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

  if (world.rank() == 0) {
    count_rows = (int)*taskData->inputs[1];
    size_rows = (int)*taskData->inputs[2];
    if (taskData->inputs_count[0] % world.size() == 0) {
      delta_proc = taskData->inputs_count[0] / world.size();
    } else {
      delta_proc = taskData->inputs_count[0] / world.size() + 1;
    }
    input_ = std::vector<int>(delta_proc * world.size(), INT_MAX);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  }
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs.size() == 3 && taskData->inputs_count.size() == 3 && taskData->outputs.size() == 1 &&
           taskData->outputs_count.size() == 1 && *taskData->inputs[1] != 0 && *taskData->inputs[2] != 0 &&
           *taskData->inputs[1] == taskData->outputs_count[0];
  }
  return true;
}

bool kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  broadcast(world, count_rows, 0);
  broadcast(world, size_rows, 0);
  broadcast(world, delta_proc, 0);

  local_input_ = std::vector<int>(delta_proc);
  boost::mpi::scatter(world, input_.data(), local_input_.data(), delta_proc, 0);

  res = std::vector<int>(count_rows, INT_MAX);

  unsigned int last_delta = 0;
  if (world.rank() == world.size() - 1) {
    last_delta = local_input_.size() * world.size() - size_rows * count_rows;
  }

  unsigned int ind = std::min(world.rank() * local_input_.size() / size_rows, static_cast<size_t>(count_rows - 1));

  unsigned int delta = std::min(local_input_.size(), size_rows - world.rank() * local_input_.size() % size_rows);
  std::vector<int> local_res(count_rows, INT_MAX);

  local_res[ind] = *std::min_element(local_input_.begin(), local_input_.begin() + delta);
  ++ind;

  unsigned int k = 0;
  while (local_input_.begin() + delta + k * size_rows < local_input_.end() - last_delta) {
    local_res[ind] =
        *std::min_element(local_input_.begin() + delta + k * size_rows,
                          std::min(local_input_.end(), local_input_.begin() + delta + (k + 1) * size_rows));
    ++k;
    ++ind;
  }

  for (unsigned int i = 0; i < res.size(); ++i) {
    reduce(world, local_res[i], res[i], boost::mpi::minimum<int>(), 0);
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
