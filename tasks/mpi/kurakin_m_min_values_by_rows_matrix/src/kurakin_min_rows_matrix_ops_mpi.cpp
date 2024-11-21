#include "mpi/kurakin_m_min_values_by_rows_matrix/include/kurakin_min_rows_matrix_ops_mpi.hpp"

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

  if (world.rank() == 0) {
    count_rows = (int)*taskData->inputs[1];
    size_rows = (int)*taskData->inputs[2];
    if (count_rows % world.size() == 0) {
      delta_rows = count_rows / world.size();
      delta_input = delta_rows * size_rows;
    } else {
      delta_rows = count_rows / world.size() + 1;
      delta_input = delta_rows * size_rows;
    }
    input_ = std::vector<int>(delta_input * world.size(), INT_MAX);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }
  }

  broadcast(world, count_rows, 0);
  broadcast(world, size_rows, 0);
  broadcast(world, delta_input, 0);
  broadcast(world, delta_rows, 0);
  

  local_input_ = std::vector<int>(delta_input);
  
  if (world.rank() == 0) {
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta_input);
    for (int i = 1; i < world.size(); i++) {
      world.send(i, 0, input_.data() + delta_input * i, delta_input);
    }
  } else {
    world.recv(0, 0, local_input_.data(), delta_input);
  }

  res = std::vector<int>(delta_rows * world.size(), INT_MAX);
  std::vector<int> local_res(delta_rows * world.size(), INT_MAX);
  for (int i = 0; i < delta_rows; i++) {
    local_res[i + world.rank() * delta_rows] =
        *std::min_element(local_input_.begin() + i * size_rows, local_input_.begin() + (i + 1) * size_rows);
  }

  for (int i = 0; i < delta_rows * world.size(); ++i) {
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