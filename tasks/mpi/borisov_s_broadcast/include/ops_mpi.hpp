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

namespace borisov_s_broadcast {

std::vector<double> getRandomPoints(int count);

class DistanceMatrixTaskSequential : public ppc::core::Task {
 public:
  explicit DistanceMatrixTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> points_;
  std::vector<double> distance_matrix_;
};

class DistanceMatrixTaskParallel : public ppc::core::Task {
 public:
  explicit DistanceMatrixTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> points_;
  std::vector<double> local_points_;
  std::vector<double> distance_matrix_;
  boost::mpi::communicator world;
};

}  // namespace borisov_s_broadcast