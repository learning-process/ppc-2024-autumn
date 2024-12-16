// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <cmath>
#include <functional>
#include <vector>
#include <string>
#include <utility>

#include "core/task/include/task.hpp"

namespace plekhanov_d_trapez_integration_mpi {

class TrapezoidalIntegralSequential : public ppc::core::Task {
 public:
  explicit TrapezoidalIntegralSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  void set_function(const std::function<double(double)>& f);

 private:
  double a_{}, b_{}, n_{}, res_{};
  std::function<double(double)> function_;

  static double integrate_function(double a, double b, int n, const std::function<double(double)>& f);
};

class TrapezoidalIntegralParallel : public ppc::core::Task {
 public:
  explicit TrapezoidalIntegralParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  void set_function(const std::function<double(double)>& f);

 private:
  double a_{}, b_{}, n_{}, res_{};
  std::function<double(double)> function_;
  boost::mpi::communicator world;

  double integrate_function(double a, double b, int n, const std::function<double(double)>& f);
};
}  // namespace plekhanov_d_trapez_integration_mpi