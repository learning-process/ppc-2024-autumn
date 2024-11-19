#pragma once

#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korablev_v_jacobi_method_mpi {

class JacobiMethodSequential : public ppc::core::Task {
 public:
  explicit JacobiMethodSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  size_t n;

  size_t maxIterations_ = 2000;
  double epsilon_ = 1e-5;
  bool isNeedToComplete(const std::vector<double>& x_old, const std::vector<double>& x_new) const;
};

class JacobiMethodParallel : public ppc::core::Task {
 public:
  explicit JacobiMethodParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  std::vector<double> x_prev;
  size_t n;

  std::vector<double> local_A;
  std::vector<double> local_b;

  std::vector<int> sizes_a;
  std::vector<int> displs_a;
  std::vector<int> sizes_b;
  std::vector<int> displs_b;

  size_t maxIterations_ = 2000;
  double epsilon_ = 1e-5;
  bool isNeedToComplete(const std::vector<double>& x_old, const std::vector<double>& x_new) const;

  boost::mpi::communicator world;
  static void calculate_distribution_a(int rows, int num_proc, std::vector<int>& sizes, std::vector<int>& displs);
  static void calculate_distribution_b(int len, int num_proc, std::vector<int>& sizes, std::vector<int>& displs);
};

}  // namespace korablev_v_jacobi_method_mpi
