#pragma once
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace smirnov_i_integration_by_rectangles {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_function(double (*func)(double));

 private:
  double res{};
  double left{};
  double right{};
  int n_{};
  double seq_integrate_rect(double (*func)(double), double left_, double right_, int n);
  double (*f)(double) = nullptr;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_function(double (*func)(double));

 private:
  double glob_res{};
  double left{};
  double right{};
  int n_{};
  boost::mpi::communicator world;
  double mpi_integrate_rect(double (*func)(double), double left_, double right_, int n);
  double (*f)(double) = nullptr;
};
}  // namespace smirnov_i_integration_by_rectangles
