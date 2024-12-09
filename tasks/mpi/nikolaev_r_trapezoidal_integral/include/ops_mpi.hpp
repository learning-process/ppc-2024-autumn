#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace nikolaev_r_trapezoidal_integral_mpi {

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
}  // namespace nikolaev_r_trapezoidal_integral_mpi