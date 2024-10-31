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

namespace gnitienko_k_sum_row_mpi {

std::vector<int> getRandomVector(int sz);

class SumByRowMPISeq : public ppc::core::Task {
 public:
  explicit SumByRowMPISeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> mainFunc();
  std::vector<int> input_{}, res{};
  int rows{}, cols{};
};

class SumByRowMPIParallel : public ppc::core::Task {
 public:
  explicit SumByRowMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> mainFunc(int StartRow, int LastRow);
  std::vector<int> input_{}, res{};
  int rows{}, cols{};
  boost::mpi::communicator world;
};

}  // namespace gnitienko_k_sum_row_mpi