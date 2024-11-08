#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace korablev_v_jacobi_method_seq {

class JacobiMethodSequential : public ppc::core::Task {
 public:
  explicit JacobiMethodSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> A_;
  std::vector<double> b_;
  std::vector<double> x_;
  int maxIterations_ = 2000;
  double epsilon_ = 1e-8;
};

}  // namespace korablev_v_jacobi_method_seq