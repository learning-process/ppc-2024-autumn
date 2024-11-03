#ifndef OPS_MPI_HPP
#define OPS_MPI_HPP

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace belov_a_max_value_of_matrix_elements_mpi {

template <typename T>
class MaxValueOfMatrixElementsParallel : public ppc::core::Task {
 public:
  explicit MaxValueOfMatrixElementsParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;

  int rows_ = 0;
  int cols_ = 0;

  T global_max_{};
  T local_max_ = std::numeric_limits<T>::lowest();
  std::vector<T> matrix;

  static T get_max_matrix_element(const std::vector<T>& matrix);
};

template <typename T>
T MaxValueOfMatrixElementsParallel<T>::get_max_matrix_element(const std::vector<T>& matrix) {
  return matrix.empty() ? 0 : *std::max_element(matrix.begin(), matrix.end());
}

template <typename T>
bool MaxValueOfMatrixElementsParallel<T>::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* dimensions = reinterpret_cast<int*>(taskData->inputs[0]);
    rows_ = dimensions[0];
    cols_ = dimensions[1];
    auto* inputMatrixData = reinterpret_cast<T*>(taskData->inputs[1]);
    matrix.assign(inputMatrixData, inputMatrixData + rows_ * cols_);
  }

  return true;
}

template <typename T>
bool MaxValueOfMatrixElementsParallel<T>::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return !taskData->inputs.empty() && reinterpret_cast<int*>(taskData->inputs[0])[0] > 0 &&
           reinterpret_cast<int*>(taskData->inputs[0])[1] > 0;
  }
  return true;
}

template <typename T>
bool MaxValueOfMatrixElementsParallel<T>::run() {
  internal_order_test();

  int rank = world.rank();
  int size = world.size();

  boost::mpi::broadcast(world, rows_, 0);
  boost::mpi::broadcast(world, cols_, 0);

  int delta = rows_ * cols_ / size;
  int remainder = rows_ * cols_ % size;

  std::vector<int> distr(size, delta);
  std::vector<int> displ(size, 0);

  for (int i = 0; i < remainder; ++distr[i], ++i);
  for (int i = 1; i < size; ++i) {
    displ[i] = displ[i - 1] + distr[i - 1];
  }

  std::vector<T> local_matrix(distr[rank]);

  if (rank == 0) {
    boost::mpi::scatterv(world, matrix, distr, displ, local_matrix.data(), distr[0], 0);
  } else {
    boost::mpi::scatterv(world, local_matrix.data(), distr[rank], 0);
  }

  local_max_ = get_max_matrix_element(local_matrix);
  boost::mpi::reduce(world, local_max_, global_max_, boost::mpi::maximum<T>(), 0);

  return true;
}

template <typename T>
bool MaxValueOfMatrixElementsParallel<T>::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<T*>(taskData->outputs[0])[0] = global_max_;
  }

  return true;
}

template <typename T>
class MaxValueOfMatrixElementsSequential : public ppc::core::Task {
 public:
  explicit MaxValueOfMatrixElementsSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rows_ = 0;
  int cols_ = 0;
  T res{};
  std::vector<T> matrix;

  static T get_max_matrix_element(const std::vector<T>& matrix);
};

template <typename T>
T MaxValueOfMatrixElementsSequential<T>::get_max_matrix_element(const std::vector<T>& matrix) {
  return *std::max_element(matrix.begin(), matrix.end());
}

template <typename T>
bool MaxValueOfMatrixElementsSequential<T>::pre_processing() {
  internal_order_test();

  auto* dimensions = reinterpret_cast<int*>(taskData->inputs[0]);
  rows_ = dimensions[0];
  cols_ = dimensions[1];

  auto* inputMatrixData = reinterpret_cast<T*>(taskData->inputs[1]);
  matrix.assign(inputMatrixData, inputMatrixData + rows_ * cols_);

  return true;
}

template <typename T>
bool MaxValueOfMatrixElementsSequential<T>::validation() {
  internal_order_test();

  return !taskData->inputs.empty() && reinterpret_cast<int*>(taskData->inputs[0])[0] > 0 &&
         reinterpret_cast<int*>(taskData->inputs[0])[1] > 0;
}

template <typename T>
bool MaxValueOfMatrixElementsSequential<T>::run() {
  internal_order_test();

  res = get_max_matrix_element(matrix);
  return true;
}

template <typename T>
bool MaxValueOfMatrixElementsSequential<T>::post_processing() {
  internal_order_test();

  reinterpret_cast<T*>(taskData->outputs[0])[0] = res;
  return true;
}

}  // namespace belov_a_max_value_of_matrix_elements_mpi

#endif  // OPS_MPI_HPP