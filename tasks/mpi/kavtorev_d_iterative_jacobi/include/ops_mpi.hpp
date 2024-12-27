#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kavtorev_d_iterative_jacobi_mpi {

bool jacobi_method(int N, const std::vector<double>& A_flat, const std::vector<double>& F, std::vector<double>& X,
                   double eps, int iterations);

class IterativeJacobiParallelMPI : public ppc::core::Task {
 public:
  explicit IterativeJacobiParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_flat;
  std::vector<double> F;
  std::vector<double> result_vector;

  std::vector<double> local_A_flat;
  std::vector<double> local_F;
  std::vector<double> X;
  std::vector<int> sizes;
  std::vector<int> displs;
  int n;
  double eps;
  int iterations;
  int num_proc;
  int rank;

  int local_size;
  int local_displ;

  std::vector<int> sendcounts_A;
  std::vector<int> displs_A;
  std::vector<int> sendcounts_F;
  std::vector<int> displs_F;

  boost::mpi::communicator world;
};

class IterativeJacobiSequentialMPI : public ppc::core::Task {
 public:
  explicit IterativeJacobiSequentialMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_flat;
  std::vector<double> F;
  std::vector<double> result_vector;
  int n;
  double eps;
  int iterations;
};

}  // namespace kavtorev_d_iterative_jacobi_mpi
