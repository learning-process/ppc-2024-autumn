// Copyright 2024 Sedova Olga
#include "mpi/sedova_o_max_of_vector_elements/include/ops_mpi.hpp"

#include <mpi.h>

#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> sedova_o_max_of_vector_elements_mpi::generate_random_vector(size_t size, size_t value) {
  std::random_device dev;
  std::mt19937 random(dev());
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = random() % (value + 1);
  }
  return vec;
}
std::vector<std::vector<int>> sedova_o_max_of_vector_elements_mpi::generate_random_matrix(size_t rows, size_t cols,
                                                                                          size_t value) {
  std::vector<std::vector<int>> matrix(rows);
  for (size_t i = 0; i < rows; i++) {
    matrix[i] = sedova_o_max_of_vector_elements_mpi::generate_random_vector(cols, value);
  }
  return matrix;
}

int sedova_o_max_of_vector_elements_mpi::find_max_of_matrix(const std::vector<int> matrix) {
  int max = matrix[0];
  for (int i = 0; i < matrix.size(); i++) {
    if (matrix[i] > max) {
      max = matrix[i];
    }
  }
  return max;
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
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] == 1;
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
  return (world.rank() != 0) || ((taskData->outputs_count[0] == 1) && (!taskData->inputs.empty()));
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  // Get rank and size of the communicator
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Calculate chunk size and boundaries
  int chunk_size = input_.size() / size;
  int start = rank * chunk_size;
  int end = (rank == size - 1) ? input_.size() : start + chunk_size;

  // Find local maximum
  auto max_it = std::max_element(input_.begin() + start, input_.begin() + end);
  int local_max = *(max_it);

  // Use MPI_Reduce to find global maximum
  int global_max;
  MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  res_ = global_max;

  return true;
}

bool sedova_o_max_of_vector_elements_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    reinterpret_cast<int *>(taskData->outputs[0])[0] = res_;
  }
  return true;
}
