// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korotin_e_multidimentional_integrals_monte_carlo_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  double possible_error();

 private:
  std::vector<std::pair<double, double>> input_;
  std::vector<double> rng;
  double res{};
  int dim;
  size_t N;
  double variance{};
  double M{};
  double (*f)(double*, int) = nullptr;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  double possible_error();

 private:
  std::vector<double> input_left_, input_right_;
  std::vector<double> rng;
  double res{};
  int dim;
  size_t N, n;
  double variance, local_variance;
  double M, local_M;
  double (*f)(double*, int) = nullptr;
  boost::mpi::communicator world;
};

}  // namespace korotin_e_multidimentional_integrals_monte_carlo_mpi
