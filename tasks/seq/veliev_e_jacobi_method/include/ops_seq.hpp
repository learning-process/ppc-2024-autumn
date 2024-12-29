// Copyright 2023 Nesterov Alexander
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace veliev_e_jacobi_method {

class MethodJacobi : public ppc::core::Task {
 public:
  explicit MethodJacobi(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int N{};
  double eps{};
  std::vector<double> matrixA;
  std::vector<double> rshB;
  std::vector<double> initialGuessX;
  void iteration_J();
};

}  // namespace veliev_e_jacobi_method
