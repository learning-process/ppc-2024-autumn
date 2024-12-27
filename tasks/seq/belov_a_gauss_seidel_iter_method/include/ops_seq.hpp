#ifndef OPS_SEQ_HPP
#define OPS_SEQ_HPP

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace belov_a_gauss_seidel_seq {

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

}  // namespace belov_a_gauss_seidel_seq

#endif  // OPS_SEQ_HPP