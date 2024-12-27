#ifndef OPS_MPI_HPP
#define OPS_MPI_HPP

#include <mpi.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <limits>
#include <vector>

#include "core/task/include/task.hpp"

namespace belov_a_gauss_seidel_mpi {

class GaussSeidelParallel : public ppc::core::Task {
 public:
  explicit GaussSeidelParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static bool ge_double(double a, double b);

 private:
  boost::mpi::communicator world;
  int n{};
  double epsilon{};
  double norm{};
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> x;
};

class GaussSeidelSequential : public ppc::core::Task {
 public:
  explicit GaussSeidelSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  static bool ge_double(double a, double b);

 private:
  int n = 0;  // matrix size
  double epsilon{};
  std::vector<double> A;  // square matrix of SLAE coeffs
  std::vector<double> b;  // free members vector
  std::vector<double> x;  // vector of initial guess for unknowns
};

bool isDiagonallyDominant(const std::vector<double>&, int n);

}  // namespace belov_a_gauss_seidel_mpi

#endif  // OPS_MPI_HPP