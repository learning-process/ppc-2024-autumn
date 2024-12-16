// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace zolotareva_a_SLE_gradient_method_mpi {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static std::vector<double> conjugate_gradient(const std::vector<double>& A, const std::vector<double>& b,
                                                std::vector<double>& x, int N);
  static double dot_product(const std::vector<double>& vec1, const std::vector<double>& vec2, int n);
  static std::vector<double> matrix_vector_mult(const std::vector<double>& matrix, const std::vector<double>& vector,
                                                int n);
  static bool is_positive_definite(const std::vector<double>& A, int n);

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  int n_{0};
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> X_;
  std::vector<double> local_A_;
  std::vector<double> local_b_;
  std::vector<double> x_;
  int n_{0};
  int local_rows{0};
  boost::mpi::communicator world;
};

}  // namespace zolotareva_a_SLE_gradient_method_mpi