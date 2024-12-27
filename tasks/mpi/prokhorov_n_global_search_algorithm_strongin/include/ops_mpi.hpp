// Copyright 2023 Nesterov Alexander
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

namespace prokhorov_n_global_search_algorithm_strongin_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double a{};
  double b{};
  double epsilon{};
  double result{};

  std::function<double(double)> f;

  double stronginAlgorithm(double a_, double b_, double epsilon_);
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double a{};
  double b{};
  double epsilon{};
  double result{};

  std::function<double(double)> f;
  boost::mpi::communicator world;

  double stronginAlgorithm(double a, double b, double epsilon);
  double stronginAlgorithmParallel(double a_, double b_, double epsilon_);
};

}  // namespace prokhorov_n_global_search_algorithm_strongin_mpi