// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace ermilova_d_Shell_sort_simple_merge_mpi {

std::vector<int> ShellSort(std::vector<int> &vec, const std::function<bool(int, int)> &comp);
std::vector<int> merge(std::vector<int> &vec1, std::vector<int> &vec2, const std::function<bool(int, int)> &comp);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> res;
  bool is_descending{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  std::vector<int> res;
  bool is_descending{};
  boost::mpi::communicator world;
};

}  // namespace ermilova_d_Shell_sort_simple_merge_mpi