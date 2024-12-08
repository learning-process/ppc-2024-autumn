#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_iterative_seidel_method_mpi {
void getRandomSLAU(std::vector<double>& A, std::vector<double>& b, int N);

class seidel_method_seq : public ppc::core::Task {
 public:
  explicit seidel_method_seq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  static int rank(std::vector<double> A_, int rows, int cols);
  double EPSILON;
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> result;
  int maxIterations, n;
};

class seidel_method_mpi : public ppc::core::Task {
 public:
  explicit seidel_method_mpi(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  static int rank(std::vector<double> A_, int rows, int cols);
  boost::mpi::communicator world;
  double EPSILON;
  std::vector<double> A, local_A;
  std::vector<double> b, local_b;
  std::vector<double> x, prevX, local_x, result;
  int maxIterations, n;
};
}  // namespace khokhlov_a_iterative_seidel_method_mpi