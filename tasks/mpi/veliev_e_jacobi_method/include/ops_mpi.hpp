#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace veliev_e_jacobi_method_mpi {

using boost::mpi::communicator;

int rankOfMatrix(std::vector<double>& matrix, int n);
bool hasUniqueSolution(std::vector<double>& A, std::vector<double>& b, int n);

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
  communicator world;
};

}  // namespace veliev_e_jacobi_method_mpi
