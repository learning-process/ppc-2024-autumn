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

namespace grudzin_k_all_reduce_boost_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int colums{};
  int rows{};
  std::vector<int> input_{};
  std::vector<int> res_{};
  std::vector<int> cnt_{};
};

class TestMPITaskBoostRealization : public ppc::core::Task {
 public:
  explicit TestMPITaskBoostRealization(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int colums{};
  int rows{};
  std::vector<int> input_{};
  std::vector<int> local_input_{};
  std::vector<int> res_{};
  std::vector<int> cnt_{};
  boost::mpi::communicator world;
};
}  // namespace grudzin_k_all_reduce_boost_mpi