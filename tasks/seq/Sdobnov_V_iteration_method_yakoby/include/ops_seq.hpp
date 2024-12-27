// Copyright 2024 Sdobnov Vladimir
#pragma once
#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"

namespace Sdobnov_iteration_method_yakoby_seq {

std::vector<double> iteration_method_yakoby(int n, const std::vector<double>& A, const std::vector<double>& b,
                                            double tolerance = 1e-6, int maxIterations = 1000);

class IterationMethodYakobySeq : public ppc::core::Task {
 public:
  explicit IterationMethodYakobySeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> matrix_;
  std::vector<double> free_members_;
  std::vector<double> res_;
  int size_;
  double tolerance = 1e-6;
  int maxIterations = 100;
};
}  // namespace Sdobnov_iteration_method_yakoby_seq