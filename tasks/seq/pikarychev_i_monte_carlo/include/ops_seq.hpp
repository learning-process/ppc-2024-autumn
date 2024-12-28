// Copyright 2023 Nesterov Alexander
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace pikarychev_i_monte_carlo_seq {

double function_double(double x);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  double a;
  double b;
  int num_samples;
  int seed;
  double range_width;
  double res;
};

}  // namespace pikarychev_i_monte_carlo_seq