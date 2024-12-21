#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kavtorev_d_iterative_jacobi_seq {

bool jacobi_method(int N, const std::vector<double>& A_flat, const std::vector<double>& F, std::vector<double>& X,
                   double eps, int iterations);

class IterativeJacobiSequential : public ppc::core::Task {
 public:
  explicit IterativeJacobiSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_flat;
  std::vector<double> F;
  std::vector<double> result_vector;
  int n;
  double eps;
  int iterations;
};

}  // namespace kavtorev_d_iterative_jacobi_seq
