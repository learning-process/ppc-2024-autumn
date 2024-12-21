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

namespace suvorov_d_shell_with_ord_merge_mpi {

std::vector<int> shell_sort(const std::vector<int>&);

class TaskShellSortSeq : public ppc::core::Task {
 public:
  explicit TaskShellSortSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> data_to_sort;
  std::vector<int> sorted_data;
};

class TaskShellSortParallel : public ppc::core::Task {
 public:
  explicit TaskShellSortParallel(std::shared_ptr<ppc::core::TaskData> taskData_, std::string ops_)
      : Task(std::move(taskData_)), ops(std::move(ops_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> data_to_sort, sorted_data, partial_data;
  boost::mpi::communicator world;
};

}  // namespace suvorov_d_shell_with_ord_merge_seq