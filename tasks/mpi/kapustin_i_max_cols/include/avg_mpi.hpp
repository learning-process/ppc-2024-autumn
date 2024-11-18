#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kapustin_i_max_column_task_mpi {
std::vector<int> getRandomVector(int sz);

class MaxColumnTaskSequentialMPI : public ppc::core::Task {
 public:
  explicit MaxColumnTaskSequentialMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> res, input_;
  int row_count{}, column_count{};
};

class MaxColumnTaskParallelMPI : public ppc::core::Task {
 public:
  explicit MaxColumnTaskParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int row_count{}, column_count{}, column_per_proc, start_current_column, end_current_column;
  std::vector<int> input_, res, gathered_max_columns, columns_per_process_count;
  boost::mpi::communicator world;
};

}  // namespace kapustin_i_max_column_task_mpi