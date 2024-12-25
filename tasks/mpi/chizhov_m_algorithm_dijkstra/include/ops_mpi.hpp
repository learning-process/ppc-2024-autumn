// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace chizhov_m_dijkstra_mpi {

void generateMatrix(std::vector<int>& w, int n, int min, int max);

void convertToCRS(const std::vector<int>& w, std::vector<int>& values, std::vector<int>& colIndex,
                  std::vector<int>& rowPtr, int n);

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
  int st{};
  int size{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  std::vector<int> res_;
  std::vector<int> values;
  std::vector<int> colIndex;
  std::vector<int> rowPtr;
  int st{};
  int size{};
  boost::mpi::communicator world;
};

}  // namespace chizhov_m_dijkstra_mpi