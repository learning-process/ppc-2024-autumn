// Copyright 2024 Sedova Olga
#include "mpi/sedova_o_max_of_vector_elements/include/ops_mpi.hpp"

#include <mpi.h>

#include <random>

int sedova_o_max_of_vector_elements_mpi::find_max_of_matrix(std::vector<int> &matrix) {
  if (matrix.empty()) {
    return std::numeric_limits<int>::min();
  }
  auto max_it = std::max_element(matrix.begin(), matrix.end());
  return *max_it;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<int>(taskData->inputs_count[0] * taskData->inputs_count[1]);
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto *input_data = reinterpret_cast<int *>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      input_[i * taskData->inputs_count[1] + j] = input_data[j];
    }
  }
  return true;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 1 && taskData->inputs_count[1] >= 1 && taskData->outputs_count[0] == 1;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  res_ = sedova_o_max_of_vector_elements_mpi::find_max_of_matrix(input_);
  return true;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = res_;
  return true;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    unsigned int rows = taskData->inputs_count[0];
    unsigned int cols = taskData->inputs_count[1];

    input_ = std::vector<int>(rows * cols);

    for (unsigned int i = 0; i < rows; i++) {
      auto *input_data = reinterpret_cast<int *>(taskData->inputs[i]);
      for (unsigned int j = 0; j < cols; j++) {
        input_[i * cols + j] = input_data[j];
      }
    }
  }

  return true;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  return (world.rank() != 0) ||
         ((taskData->outputs_count[0] == 1) && (taskData->inputs_count[0] > 0) && (!taskData->inputs.empty()));
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int a = 0;
  if (world.rank() == 0) {
    unsigned int rows = taskData->inputs_count[0];
    unsigned int cols = taskData->inputs_count[1];
    a = rows * cols / world.size();
    int b = rows * cols % world.size() + 1;
    if (a == 0) {
      for (int i = 1; i < world.size(); i++) {
        world.send(i, 0, 0);
      }
      linput_ = std::vector<int>(input_.begin(), input_.begin() + b - 1);
      res_ = sedova_o_max_of_vector_elements_mpi::find_max_of_matrix(linput_);
      return true;
    }
    for (int i = 1; i < world.size(); i++) {
      world.send(i, 0, a + (int)(i < b));
    }
    for (int i = 1; i < b; i++) {
      world.send(i, 0, input_.data() + a * i + i - 1, a + 1);
    }
    for (int i = b; i < world.size(); i++) {
      world.send(i, 0, input_.data() + a * i + b - 1, a);
    }
    linput_ = std::vector<int>(input_.begin(), input_.begin() + a);
  }

  if (world.rank() != 0) {
    world.recv(0, 0, a);
    if (a == 0) {
      return true;
    }
    linput_ = std::vector<int>(a);
    world.recv(0, 0, input_.data(), a);
  }

  int lres_ = sedova_o_max_of_vector_elements_mpi::find_max_of_matrix(input_);
  reduce(world, lres_, res_, boost::mpi::maximum<int>(), 0);
  return true;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = res_;
  }
  return true;
}
