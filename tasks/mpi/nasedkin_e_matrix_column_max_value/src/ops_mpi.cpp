// Copyright 2023 Nasedkin Egor
#include "mpi/nasedkin_e_matrix_column_max_value/include/ops_mpi.hpp"

#include <algorithm>
#include <random>
#include <vector>

namespace nasedkin_e_matrix_column_max_value_mpi {

std::vector<int> getRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool MatrixColumnMaxMPI::pre_processing() {
  internal_order_test();
  int rows = taskData->inputs_count[0] / taskData->inputs_count[1];
  int cols = taskData->inputs_count[1];
  input_ = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);

  // Проверка на корректность значения taskData->inputs_count[0]
  if (taskData->inputs_count[0] <= 0) {
    // Выполните соответствующие действия, например, выброс исключения или возврат ошибки
    throw std::invalid_argument("Invalid input count");
  }

  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_[i] = tmp_ptr[i];
  }

  int delta = rows / world.size();
  local_input_ = std::vector<int>(delta * cols);
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.UninitializedObject)
  if (world.rank() == 0) {
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta * cols, delta * cols);
    }
    local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta * cols);
  } else {
    world.recv(0, 0, local_input_.data(), delta * cols);
  }

  res_ = std::vector<int>(cols, 0);
  return true;
}

bool MatrixColumnMaxMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == taskData->inputs_count[1];
  }
  return true;
}

bool MatrixColumnMaxMPI::run() {
  internal_order_test();
  int rows = local_input_.size() / res_.size();
  for (size_t col = 0; col < res_.size(); col++) {
    int local_max = local_input_[col];
    for (int row = 1; row < rows; row++) {
      if (local_input_[row * res_.size() + col] > local_max) {
        local_max = local_input_[row * res_.size() + col];
      }
    }
    reduce(world, local_max, res_[col], boost::mpi::maximum<int>(), 0);
  }
  return true;
}

bool MatrixColumnMaxMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
    for (unsigned i = 0; i < res_.size(); i++) {
      tmp_ptr[i] = res_[i];
    }
  }
  return true;
}

}  // namespace nasedkin_e_matrix_column_max_value_mpi