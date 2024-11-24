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

namespace chizhov_m_all_reduce_boost_mpi {

std::vector<int> getRandomVector(int sz);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_{};
  int cols{};
  int rows{};
  std::vector<int> sum{};
};

class TestMPITaskBoostParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskBoostParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_{};
  std::vector<int> local_input_{};
  std::vector<int> res_{};
  int cols{};
  int rows{};
  std::vector<int> sum{};
  boost::mpi::communicator world;
};

}  // namespace chizhov_m_all_reduce_mpi