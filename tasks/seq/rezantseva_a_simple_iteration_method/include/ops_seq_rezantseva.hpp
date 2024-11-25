// Copyright 2023 Nesterov Alexander
#pragma once
#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace rezantseva_a_simple_iteration_method_seq {

std::pair<std::vector<double>, std::vector<double>> createRandomMatrix(size_t n);

class SimpleIterationSequential : public ppc::core::Task {
 public:
  explicit SimpleIterationSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A_;      // coefficient matrix
  std::vector<double> b_;      // free members vector
  std::vector<double> x_;      // current approach
  double epsilon_ = 1e-3;      // precision
  size_t maxIteration_ = 100;  // to avoid endless cycle
  bool checkMatrix();          // we check convergence condition (|A11| > |A12| + |A13| + .. + |A1n|) etc
  bool isTimeToStop(const std::vector<double>& x0,
                    const std::vector<double>& x1) const;  // stop if |xn^(i+1) - xn^i| < epsilon
};

}  // namespace rezantseva_a_simple_iteration_method_seq