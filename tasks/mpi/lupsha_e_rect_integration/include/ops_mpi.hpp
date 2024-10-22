// Copyright 2024 Lupsha Egor
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace lupsha_e_rect_integration_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void function_set(const std::function<double(double)>& func);

 private:
  double lower_bound{};
  double upper_bound{};
  int num_intervals{};
  std::function<double(double)> f;
  std::vector<double> input_;
  std::vector<double> results_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void function_set(const std::function<double(double)>& func);

 private:


  double integrate(const std::function<double(double)>& f_, double lower_bound_, double upper_bound_, 
	               int num_intervals_);
  double lower_bound{};
  double upper_bound{};
  double local_sum_{};
  double global_sum_{};
  int num_intervals{};
  std::function<double(double)> f;
  std::vector<double> input_;
  std::vector<double> results_;
  boost::mpi::communicator world;
};

}  // namespace lupsha_e_rect_integration_mpi