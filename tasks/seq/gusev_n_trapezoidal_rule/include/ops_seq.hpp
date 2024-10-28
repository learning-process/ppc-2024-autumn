// Copyright 2024 Nesterov Alexander
#pragma once

#include <vector>
#include <functional>
#include <memory>
#include "core/task/include/task.hpp"

namespace gusev_n_trapezoidal_rule_seq {

class TrapezoidalIntegrationSequential : public ppc::core::Task {
 public:
  explicit TrapezoidalIntegrationSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void set_function(const std::function<double(double)>& func);

 private:
  static double integrate(const std::function<double(double)>& f, double a, double b, int n);

  double a_{};
  double b_{};
  int n_{};
  double result_{};
  std::function<double(double)> func_;
};

}  // namespace gusev_n_trapezoidal_rule