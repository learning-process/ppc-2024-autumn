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

namespace kondratev_ya_max_col_matrix_mpi {

std::vector<std::vector<int32_t>> getRandomMatrix(uint32_t row, uint32_t col);
void insertRefValue(std::vector<std::vector<int32_t>>& mtrx, int32_t ref);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int32_t>> input_;
  std::vector<int32_t> res_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int32_t>> input_;
  std::vector<std::vector<int32_t>> local_input_;
  std::vector<int32_t> res_;

  uint32_t row_;
  uint32_t col_;
  uint32_t step_;
  uint32_t remain_;
  boost::mpi::communicator world;
};

}  // namespace kondratev_ya_max_col_matrix_mpi