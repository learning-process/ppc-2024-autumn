#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_monte_carlo_integration {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  double a = 0.0;
  double b = 0.0;
  int num_samples = 0;
  static double function_square(double x) { return x * x; }

 private:
  double res{};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  double a = 0.0;
  double b = 0.0;
  int num_samples = 0;
  int local_num_samples = 0;
  static double function_square(double x) { return x * x; }

 private:
  double res;
  boost::mpi::communicator world;
};

}  // namespace malyshev_v_monte_carlo_integration
