// Copyright 2023 Nesterov Alexander
#include "mpi/kovalchuk_a_max_of_vector_elements/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <climits>
#include <limits>
#include <numeric>
#include <random>
<<<<<<< HEAD
#include <algorithm>
=======
>>>>>>> d8cdde3175d86a868290c246bfeb882051127316

namespace kovalchuk_a_max_of_vector_elements_mpi {

std::vector<int> getRandomVector(int size, int start_gen, int fin_gen) {
  static std::random_device dev;
  static std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(start_gen, fin_gen);
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

std::vector<std::vector<int>> getRandomMatrix(int rows, int columns, int start_gen, int fin_gen) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(columns));
  for (int i = 0; i < rows; i++) {
    matrix[i] = getRandomVector(columns, start_gen, fin_gen);
  }
  return matrix;
}

bool TestTaskMPI::pre_processing() {
  internal_order_test();
  inputMatrix_ = std::vector<std::vector<int>>(taskData->inputs_count[0], std::vector<int>(taskData->inputs_count[1]));
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < taskData->inputs_count[1]; j++) {
      inputMatrix_[i][j] = tmp_ptr[j];
    }
  }
  result_ = std::numeric_limits<int>::min();
  return true;
}

bool TestTaskMPI::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] == 1;
}

bool TestTaskMPI::run() {
  internal_order_test();
  result_ = std::numeric_limits<int>::min();
  int local_max = std::numeric_limits<int>::min();

  int start_row = world.rank() * inputMatrix_.size() / world.size();
  int end_row = (world.rank() + 1) * inputMatrix_.size() / world.size();

  for (int i = start_row; i < end_row; i++) {
    int row_max = *std::max_element(inputMatrix_[i].begin(), inputMatrix_[i].end());
    if (row_max > local_max) {
      local_max = row_max;
    }
  }

  boost::mpi::all_reduce(world, local_max, result_, boost::mpi::maximum<int>());

  return true;
}

bool TestTaskMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = result_;
  }
  return true;
}

}  // namespace kovalchuk_a_max_of_vector_elements_mpi