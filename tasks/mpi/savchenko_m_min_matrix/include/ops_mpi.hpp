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

namespace savchenko_m_min_matrix_mpi {

std::vector<int> getRandomMatrix(int rows, int columns, int min, int max);

class TestMPITaskSequential : public ppc::core::Task {
public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

private:
  std::vector<int> matrix{};
  int res{};
  int rows, columns;
};

class TestMPITaskParallel : public ppc::core::Task {
public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

private:
  std::vector<int> matrix, local_matrix;
  int res{};
  int rows, columns;

  boost::mpi::communicator world;
};

} // namespace savchenko_m_min_matrix_mpi