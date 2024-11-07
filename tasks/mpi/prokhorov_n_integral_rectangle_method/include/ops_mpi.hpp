// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace prokhorov_n_integral_rectangle_method_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_function(const std::function<double(double)>& func);

 private:
  static double integrate(const std::function<double(double)>& f, double left_, double right_, int n);

  double left_{};
  double right_{};
  int n{};
  double res{};
  std::function<double(double)> func_;
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
  static double parallel_integrate(const std::function<double(double)>& f, double left_, double right_, int n,
                                   const boost::mpi::communicator& world);

  double left_{};
  double right_{};
  int n{};
  double local_res{};
  double global_res{};
  std::function<double(double)> func_;
  boost::mpi::communicator world;
};

}  // namespace prokhorov_n_integral_rectangle_method_mpi
