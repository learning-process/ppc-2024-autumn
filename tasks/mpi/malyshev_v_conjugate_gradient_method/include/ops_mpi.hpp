#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_conjugate_gradient {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> matrix_;
  std::vector<double> vector_;
  std::vector<double> result_;
};

class TestTaskParallel : public ppc::core::Task {
 public:
  explicit TestTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> matrix_;
  std::vector<double> vector_;
  std::vector<double> result_;

  boost::mpi::communicator world;
};

}  // namespace malyshev_conjugate_gradient