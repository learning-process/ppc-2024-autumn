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

namespace poroshin_v_find_min_val_row_matrix_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  static std::vector<int> gen(int m, int n);  // Generate vector (matrix)
 private:
  std::vector<int> input_;  // Input vector
  std::vector<int> res;     // Result vector
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
 private:
  std::vector<int> input_;         // Input vector
  std::vector<int> local_input_;   // Local input vector
  std::vector<int> res;            // Result vector
  boost::mpi::communicator world;  // MPI communicator
};

}  // namespace poroshin_v_find_min_val_row_matrix_mpi