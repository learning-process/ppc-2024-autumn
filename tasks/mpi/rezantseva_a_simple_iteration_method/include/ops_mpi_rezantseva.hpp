// Copyright 2023 Nesterov Alexander
#pragma once
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace rezantseva_a_simple_iteration_method_mpi {
std::pair<std::vector<double>, std::vector<double>> createRandomMatrix(size_t n);

class SimpleIterationSequential : public ppc::core::Task {
 public:
  explicit SimpleIterationSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;       // coefficient matrix
  std::vector<double> b_;       // free members vector
  std::vector<double> x_;       // current approach
  std::vector<double> B_;       // transition matrix Bij = -Aij/Aii
  std::vector<double> c_;       // iteration vector  ci = bi/Aii
  size_t n_ = 0;                // size
  double epsilon_ = 1e-3;       // precision
  size_t maxIteration_ = 1000;  // to avoid endless cycle
  bool checkMatrix();           // check if matrix A diagonal dominant
  bool checkMatrixNorm();       // convergence condition : The norm of matrix B is less than 1
  bool isTimeToStop(const std::vector<double>& x0,
                    const std::vector<double>& x1) const;  // stop if |xn^(i+1) - xn^i| < epsilon
};

class SimpleIterationMPI : public ppc::core::Task {
 public:
  explicit SimpleIterationMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::vector<double> A_;       // coefficient matrix
  std::vector<double> b_;       // free members vector
  std::vector<double> x_;       // current approach
  std::vector<double> B_;       // transition matrix Bij = -Aij/Aii
  std::vector<double> c_;       // iteration vector  ci = bi/Aii
  std::vector<double> prev_x_;  // previous approach

  double epsilon_ = 1e-3;       // precision
  size_t maxIteration_ = 1000;  // to avoid endless cycle
  bool checkMatrix();           // we check convergence condition (|A11| > |A12| + |A13| + .. + |A1n|) etc
  bool isTimeToStop(const std::vector<double>& x0,
                    const std::vector<double>& x1) const;  // stop if |xn^(i+1) - xn^i| < epsilon
  size_t n_ = 0;
  std::vector<int> counts_;
  size_t num_processes_ = 0;
};

}  // namespace rezantseva_a_simple_iteration_method_mpi