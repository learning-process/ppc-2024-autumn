#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace plekhanov_d_allreduce_boost_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> inputData_;
  std::vector<int> resultData_;
  int columnCount{};
  int rowCount{};
  std::vector<int> countAboveMin_;
};

class TestMPITaskBoostParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskBoostParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> inputData_;
  std::vector<int> resultData_;
  std::vector<int> countAboveMin_;
  std::vector<int> count_greater;
  std::vector<int> localInputData_;
  int columnCount{};
  int rowCount{};
  boost::mpi::communicator world;
};

}  // namespace plekhanov_d_allreduce_boost_mpi