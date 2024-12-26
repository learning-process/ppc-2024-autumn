// Copyright 2024 Korobeinikov Arseny
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <numeric>
#include <utility>

#include "core/task/include/task.hpp"

namespace korobeinikov_a_test_task_mpi_lab_03 {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> values;
  std::vector<int> col;
  std::vector<int> RowIndex;

  std::vector<int> res;

  int size;
  int sv;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> values;
  std::vector<int> col;
  std::vector<int> RowIndex;

  std::vector<int> res;

  int size;
  int sv;

  boost::mpi::communicator world;
};

}  // namespace korobeinikov_a_test_task_mpi_lab_03