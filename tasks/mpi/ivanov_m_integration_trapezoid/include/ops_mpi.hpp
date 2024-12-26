// Copyright 2024 Ivanov Mike
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

namespace ivanov_m_integration_trapezoid_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

  void add_function(const ::std::function<double(double)>& f);

 private:
  double a_{}, b_{};
  int n_{};
  double result_{};
  std::function<double(double)> f_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void add_function(const ::std::function<double(double)>& f);

 private:
  double a_{}, b_{}, result_{};
  int n_{};
  std::function<double(double)> f_;
  boost::mpi::communicator world;
};

}  // namespace ivanov_m_integration_trapezoid_mpi