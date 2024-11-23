// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace titov_s_simple_iteration_mpi {

// Последовательная реализация
class MPISimpleIterationSequential : public ppc::core::Task {
 public:
  explicit MPISimpleIterationSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::unique_ptr<float[]> input_;
  std::unique_ptr<float[]> res_;
  float epsilon_;
  unsigned int rows_;
  unsigned int cols_;

  void transformMatrix();
};

// Параллельная реализация
class MPISimpleIterationParallel : public ppc::core::Task {
 public:
  explicit MPISimpleIterationParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> Matrix;
  std::vector<double> Values;
  std::vector<double> current;
  std::vector<double> prev;
  int Rows;
  int Cols;

  double epsilon_;

  std::vector<int> number_matrix;
  std::vector<int> offset_matrix;
  std::vector<int> number_values;
  std::vector<int> offset_values;
  std::vector<double> Matrix_l;
  std::vector<double> Values_l;
  boost::mpi::communicator world;
};

}  // namespace titov_s_simple_iteration_mpi
