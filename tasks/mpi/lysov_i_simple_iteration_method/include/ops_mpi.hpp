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

namespace lysov_i_simple_iteration_method_mpi {

class SlaeIterationTask : public ppc::core::Task {
 public:
  explicit SlaeIterationTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int input_size_ = 0;
  double tolerance_ = 0;
  std::vector<std::vector<double>> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  std::vector<std::vector<double>> B_;
  std::vector<double> g_;

  bool isDiagonallyDominant();
  bool transformSystem();
};

class SlaeIterationTaskMPI : public ppc::core::Task {
 public:
  explicit SlaeIterationTaskMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int input_size_ = 0;
  double tolerance_ = 1e-9;
  std::vector<std::vector<double>> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  boost::mpi::communicator world;
  std::vector<double> B_;
  std::vector<double> g_;

  bool isDiagonallyDominant();
  bool transformSystem();
};

}  // namespace lysov_i_simple_iteration_method_mpi