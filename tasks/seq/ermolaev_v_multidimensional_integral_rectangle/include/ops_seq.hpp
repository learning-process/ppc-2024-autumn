// Copyright 2023 Nesterov Alexander
#pragma once

#include <cmath>
#include <deque>
#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

namespace ermolaev_v_multidimensional_integral_rectangle_seq {
using function = std::function<double(std::vector<double>& args)>;

double integrateImpl(std::deque<std::pair<double, double>>& limits, std::vector<double>& args, const function& func,
                     double eps);
double integrate(std::deque<std::pair<double, double>> limits, double eps, const function& func);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, function& func_)
      : Task(std::move(taskData_)), function_(func_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::deque<std::pair<double, double>> limits_;
  function function_;
  double eps_{};
  double res_{};
};

}  // namespace ermolaev_v_multidimensional_integral_rectangle_seq