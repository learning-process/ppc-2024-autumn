#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <limits>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace vavilov_v_bellman_ford_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> row_offsets_, col_indices_, weights_;
  std::vector<int> distances_;
  int vertices_{0}, edges_count_{0}, source_{0};
  void CRS(const int* matrix);
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> row_offsets_, col_indices_, weights_;
  std::vector<int> distances_;
  int vertices_{0}, edges_count_{0}, source_{0};
  void CRS(const int* matrix);
  boost::mpi::communicator world;
};
}  // namespace vavilov_v_bellman_ford_mpi
