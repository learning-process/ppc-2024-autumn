#ifndef OPS_SEQ_HPP
#define OPS_SEQ_HPP

#include <cmath>
#include <iostream>
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

  bool isDiagonallyDominant() const;

 private:
  int n = 0;
  double epsilon{};
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> x;
};

std::vector<double> generateDiagonallyDominantMatrix(int n);

}  // namespace belov_a_gauss_seidel_seq

#endif  // OPS_SEQ_HPP