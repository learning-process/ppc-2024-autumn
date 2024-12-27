// Copyright 2023 Nesterov Alexander
#include "mpi/kovalchuk_a_max_of_vector_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <ranges>
#include <string>
#include <vector>

bool kovalchuk_a_max_of_vector_elements::TestMPITaskSequential ::pre_processing() {
  internal_order_test();
  // Инициализация векторов
  if (taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0) {
    input_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
      std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[1], input_[i].begin());
    }
  } else {
    input_ = std::vector<std::vector<int>>();
  }
  // Инициализация значения для вывода
  res_ = INT_MIN;
  return true;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskSequential ::validation() {
  internal_order_test();
  // Проверка количества элементов на выходе
  if (taskData->outputs_count[0] != 1) {
    return false;
  }
  // Проверка, что все векторы не пусты
  return std::ranges::all_of(input_, [](const auto& vec) { return !vec.empty(); });
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskSequential ::run() {
  internal_order_test();
  std::vector<int> local_res(input_.size());
  for (unsigned int i = 0; i < input_.size(); i++) {
    local_res[i] = *std::max_element(input_[i].begin(), input_[i].end());
  }
  res_ = *std::max_element(local_res.begin(), local_res.end());
  return true;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskSequential ::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  return true;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  unsigned int delta = 0;
  if (world.rank() == 0) {
    if (taskData->inputs_count[0] == 0 || taskData->inputs_count[1] == 0) {
      delta = 0;
    } else {
      delta = std::max(1u, taskData->inputs_count[0] * taskData->inputs_count[1] / world.size());
    }
    if (taskData->inputs_count[0] == 1 && taskData->inputs_count[1] == 1) {
      delta = 1;
    }
  }

  broadcast(world, delta, 0);

  if (world.rank() == 0) {
    // Init vectors
    unsigned int rows = taskData->inputs_count[0];
    unsigned int columns = taskData->inputs_count[1];
    if (rows > 0 && columns > 0) {
      input_ = std::vector<int>(rows * columns);
      for (unsigned int i = 0; i < rows; i++) {
        auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
        for (unsigned int j = 0; j < columns; j++) {
          input_[i * columns + j] = tmp_ptr[j];
        }
      }
      if (delta > 0) {
        for (int proc = 1; proc < world.size(); proc++) {
          std::span<int> buffer(input_.data() + delta * proc, delta);
          world.send(proc, 0, buffer.data(), buffer.size());
        }
      }
    } else {
      for (int proc = 1; proc < world.size(); proc++) {
        world.send<int>(proc, 0, nullptr, 0);
      }
    }
  }

  local_input_ = std::vector<int>(delta);
  if (world.rank() == 0) {
    if (!input_.empty()) {
      local_input_ = std::vector<int>(input_.begin(), input_.begin() + delta);
    }
  } else {
    world.recv(0, 0, local_input_.data(), delta);
  }

  // Init value for output
  res_ = INT_MIN;
  return true;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    if (taskData->outputs_count[0] != 1) {
      return false;
    }
    // Check that matrix is not empty
    if (taskData->inputs_count[0] == 0 || taskData->inputs_count[1] == 0) {
      return false;
    }
  }
  return true;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskParallel::run() {
  internal_order_test();
  int local_res = INT_MIN;
  if (!local_input_.empty()) {
    local_res = *std::max_element(local_input_.begin(), local_input_.end());
  }
  reduce(world, local_res, res_, boost::mpi::maximum<int>(), 0);
  return true;
}

bool kovalchuk_a_max_of_vector_elements::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}
