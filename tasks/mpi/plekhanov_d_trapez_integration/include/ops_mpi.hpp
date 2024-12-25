#pragma once
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace plekhanov_d_trapez_integration_mpi {

class trapezIntegrationSEQ : public ppc::core::Task {
 public:
  explicit trapezIntegrationSEQ(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
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
class trapezIntegrationMPI : public ppc::core::Task {
 public:
  explicit trapezIntegrationMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
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