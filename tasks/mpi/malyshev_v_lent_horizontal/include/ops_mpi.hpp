#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_lent_horizontal {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<int32_t>> matrix_;
  std::vector<int32_t> vector_;
  std::vector<int32_t> result_;
};

class TestTaskParallel {
 private:
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskData;

  uint32_t delta_;
  uint32_t ext_;
  std::vector<std::vector<int32_t>> matrix_, local_matrix_;
  std::vector<int32_t> vector_, result_, local_result_;
  std::vector<int32_t> flat_matrix_;

 public:
  explicit TestTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData) : taskData(std::move(taskData)) {}

  bool validation();
  bool pre_processing();
  bool run();
  bool post_processing();

 private:
  void internal_order_test() {}
};

}  // namespace malyshev_lent_horizontal