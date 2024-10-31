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

namespace kalyakina_a_average_value_mpi {

std::vector<int> RandomVectorWithFixSum(int sum, const int& count);

class FindingAverageMPITaskSequential : public ppc::core::Task {
 public:
  explicit FindingAverageMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_vector;
  double average_value{};
};

class FindingAverageMPITaskParallel : public ppc::core::Task {
 public:
  explicit FindingAverageMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> local_input_vector;
  int result{};
  boost::mpi::communicator world;
};

}  // namespace kalyakina_a_average_value_mpi