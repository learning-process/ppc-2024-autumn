// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace morozov_e_mult_sparse_matrix {
double scalMultOfVectors(const std::vector<double> &vA, const std::vector<double> &vB);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> convertToBasicMatrixs(
    const std::vector<double> &dA, const std::vector<int> &row_indA, const std::vector<int> &col_indA,
    const std::vector<double> &dB, const std::vector<int> &row_indB, const std::vector<int> &col_indB, int rowsA,
    int columnsA, int rowsB, int columnsB);
void convertToCCS(const std::vector<std::vector<double>> &matrix, std::vector<double> &values,
                  std::vector<int> &row_indices, std::vector<int> &col_pointers);
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> ans;
  std::vector<double> dA, dB;
  std::vector<int> row_indA, row_indB, col_indA, col_indB;
  int rowsA, rowsB, columnsA, columnsB, dA_size, dB_size, row_indA_size, row_indB_size, col_indA_size, col_indB_size;
};
template <typename T>
void printMatrix(std::vector<std::vector<T>> m);
template <typename T>
void printVector(std::vector<T> v);

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> ans;
  std::vector<double> dA, dB, local_input_A, local_input_B;
  std::vector<int> row_indA, row_indB, col_indA, col_indB, local_input_;
  int rowsA, rowsB, columnsA, columnsB, dA_size, dB_size, row_indA_size, row_indB_size, col_indA_size, col_indB_size;
  boost::mpi::communicator world;
};
}  // namespace morozov_e_mult_sparse_matrix
