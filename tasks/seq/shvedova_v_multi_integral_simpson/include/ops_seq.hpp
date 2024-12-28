#pragma once

#include <cmath>
#include <deque>
#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

namespace shvedova_v_multidimensional_integral_simpson_seq {

using FunctionType = std::function<double(std::vector<double>& args)>;
using limits = std::deque<std::pair<double, double>>;

double simpsonIntegrateImpl(limits& integrationLimits, std::vector<double>& currentArgs,
                            const FunctionType& targetFunction, double tolerance);

double simpsonIntegrate(limits integrationLimits, double tolerance, const FunctionType& targetFunction);

class MultidimensionalIntegralSequential : public ppc::core::Task {
 public:
  explicit MultidimensionalIntegralSequential(std::shared_ptr<ppc::core::TaskData> taskData)
      : Task(std::move(taskData)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  limits integrationLimits_;
  FunctionType targetFunction_;
  double tolerance_{};
  double result_{};
};

}  // namespace shvedova_v_multidimensional_integral_simpson_seq