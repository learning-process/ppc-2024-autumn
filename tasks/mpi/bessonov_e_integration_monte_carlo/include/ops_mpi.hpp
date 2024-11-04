#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace bessonov_e_integration_monte_carlo_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  double a, b;
  int num_points;
  static double exampl_func(double x) { return x * x * x; }

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
  double a, b;
  int num_points;
  static double exampl_func(double x) { return x * x * x; }

 private:
  double res;
  boost::mpi::communicator world;
};

}  // namespace bessonov_e_integration_monte_carlo_mpi