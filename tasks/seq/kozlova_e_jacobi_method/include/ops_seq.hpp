// Copyright 2023 Nesterov Alexander
#pragma once

#include <string>
#include <vector>
#include <cmath>
#include "core/task/include/task.hpp"

namespace kozlova_e_jacobi_method {

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
  std::vector<double> A;
  std::vector<double> B; 
  std::vector<double> X;
  void jacobi_iteration();
};

}