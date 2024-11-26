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

namespace chizhov_m_all_reduce_my_mpi {

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
  std::vector<int> res_;
  int cols{};
  int rows{};
  std::vector<int> sum;
};

class TestMPITaskMyOwnParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskMyOwnParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  template <typename T>
  void my_all_reduce(const boost::mpi::communicator& world, const T* in_values, T* out_values, int n);

 private:
  std::vector<int> input_;
  std::vector<int> local_input_;
  std::vector<int> res_;
  std::vector<int> sum;
  int cols{};
  int rows{};
  boost::mpi::communicator world;
};

}  // namespace chizhov_m_all_reduce_my_mpi