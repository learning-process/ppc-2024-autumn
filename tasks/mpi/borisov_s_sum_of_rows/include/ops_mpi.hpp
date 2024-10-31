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

namespace borisov_s_sum_of_rows {

std::vector<int> getRandomMatrix(size_t rows, size_t cols);

class SumOfRowsTaskSequential : public ppc::core::Task {
 public:
  explicit SumOfRowsTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int>> matrix_;
  std::vector<int> row_sums_;
};

class SumOfRowsTaskParallel : public ppc::core::Task {
 public:
  explicit SumOfRowsTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrix_, loc_matrix_;
  std::vector<int> row_sums_, loc_row_sums_;
  boost::mpi::communicator world;
  int res = 0;
};

}  // namespace borisov_s_sum_of_rows