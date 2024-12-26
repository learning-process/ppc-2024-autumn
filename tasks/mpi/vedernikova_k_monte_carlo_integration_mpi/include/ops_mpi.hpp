#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <random>
#include <utility>

#include "core/task/include/task.hpp"

namespace vedernikova_k_monte_carlo_integration_mpi {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::function<double(double, double, double)> f;

 private:
  double a_x, b_x, a_y, b_y, a_z, b_z;
  int num_points;
  double res{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  std::function<double(double, double, double)> f;

 private:
  double a_x, b_x, a_y, b_y, a_z, b_z;
  int num_points;
  double res{};
  boost::mpi::communicator world;
};

}  // namespace vedernikova_k_monte_carlo_integration_mpi