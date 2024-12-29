#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_conjugate_gradient {

class TestTaskParallel {
 public:
  explicit TestTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData)
      : world(boost::mpi::communicator()), task_data_(taskData) {}

  bool run();
  bool pre_processing();
  bool validation();
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> task_data_;
  std::vector<double> matrix_;
  std::vector<double> vector_;
  std::vector<double> result_;
};

class TestTaskSequential {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData) : task_data_(taskData) {}

  bool run();
  bool pre_processing();
  bool validation();
  bool post_processing() override;

 private:
  std::shared_ptr<ppc::core::TaskData> task_data_;
  std::vector<double> matrix_;
  std::vector<double> vector_;
  std::vector<double> result_;
};

}  // namespace malyshev_conjugate_gradient