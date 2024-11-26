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

namespace kozlova_e_jacobi_method_mpi {

class MethodJacobiSeq : public ppc::core::Task {
 public:
  explicit MethodJacobiSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  void jacobi_iteration();

 private:
  int N{};
  double eps{};
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> X;
};

class MethodJacobiMPI : public ppc::core::Task {
 public:
  explicit MethodJacobiMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  void jacobi_iteration();

 private:
  int N{};
  double eps{};
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> X;
  boost::mpi::communicator world;
};

}  // namespace kozlova_e_jacobi_method_mpi