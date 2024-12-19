// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace muhina_m_shell_sort_mpi {
std::vector<int> shellSort(const std::vector<int>& vect);
std::vector<int> merge(const std::vector<int>& left, const std::vector<int>& right);

class ShellSortMPISequential : public ppc::core::Task {
 public:
  explicit ShellSortMPISequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_;
};

class ShellSortMPIParallel : public ppc::core::Task {
 public:
  explicit ShellSortMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  std::vector<int> res_, local_res_;
  boost::mpi::communicator world_;
};

}  // namespace muhina_m_shell_sort_mpi
