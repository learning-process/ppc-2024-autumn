// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace fyodorov_m_trapezoidal_method_mpi {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::function<double(const std::vector<double>&)> func_;
  std::vector<double> lower_bounds_;
  std::vector<double> upper_bounds_;
  std::vector<int> intervals_;
  double result_{};

  static double evaluate_func_sequential(const std::function<double(const std::vector<double>&)>& func,
                                         const std::vector<double>& point);
};

// namespace fyodorov_m_trapezoidal_method_seq

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_,
                               std::function<double(const std::vector<double>&)> func)
      : Task(std::move(taskData_)), func_(std::move(func)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::function<double(const std::vector<double>)> func_;
  std::vector<double> lower_bounds_;
  std::vector<double> upper_bounds_;
  std::vector<int> intervals_;
  double result_{};
  boost::mpi::communicator world;
  static double evaluate_func_parallel(const std::function<double(const std::vector<double>&)>& func,
                                       const std::vector<double>& point);
};

}  // namespace fyodorov_m_trapezoidal_method_mpi

// namespace fyodorov_m_trapezoidal_method_mpi