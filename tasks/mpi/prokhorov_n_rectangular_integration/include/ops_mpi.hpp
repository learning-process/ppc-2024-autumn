#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace prokhorov_n_rectangular_integration_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_function(const std::function<double(double)>& func);

 private:
  static double integrate(const std::function<double(double)>& f, double lower_bound, double upper_bound, int n);

  double lower_bound_;
  double upper_bound_;
  int n_;
  double result_;
  std::function<double(double)> f_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_function(const std::function<double(double)>& func);

 private:
  double parallel_integrate(const std::function<double(double)>& f, double lower_bound, double upper_bound, int n);

  double lower_bound_;
  double upper_bound_;
  int n_;
  double result_;
  std::function<double(double)> f_;
  boost::mpi::communicator world;
};

}  // namespace prokhorov_n_rectangular_integration_mpi