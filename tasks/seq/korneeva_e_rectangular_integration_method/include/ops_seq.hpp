#pragma once

#include <cmath>
#include <functional>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"

namespace korneeva_e_rectangular_integration_method_seq {

using Function = std::function<double(const std::vector<double>& args_)>;

class RectangularIntegration : public ppc::core::Task {
 public:
  explicit RectangularIntegration(const std::shared_ptr<ppc::core::TaskData>& taskData_, Function func)
      : Task(std::move(taskData_)), integrandFunction(std::move(func)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double result;
  double epsilon;
  Function integrandFunction;
  std::vector<std::pair<double, double>> limits;

  double calculateIntegral(std::vector<double>& args_);
  static constexpr double MIN_EPSILON = 1e-6;
};

bool RectangularIntegration::pre_processing() {
  internal_order_test();

  auto* ptrInput = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
  limits.assign(ptrInput, ptrInput + taskData->inputs_count[0]);
  result = 0.0;

  epsilon = *reinterpret_cast<double*>(taskData->inputs[1]);
  if (epsilon < MIN_EPSILON) {
    epsilon = MIN_EPSILON;
  }
  return true;
}

bool RectangularIntegration::validation() {
  internal_order_test();

  bool validInput = taskData->inputs_count[0] > 0 && taskData->inputs.size() == 2;
  bool validOutput = taskData->outputs_count[0] == 1 && !taskData->outputs.empty();

  size_t numDimensions = taskData->inputs_count[0];
  bool validLimits = true;
  auto* ptrInput = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);

  for (size_t i = 0; i < numDimensions; ++i) {
    if (ptrInput[i].first > ptrInput[i].second) {
      validLimits = false;
      break;
    }
  }
  return validInput && validOutput && validLimits;
}

bool RectangularIntegration::run() {
  internal_order_test();
  std::vector<double> args_;
  result = calculateIntegral(args_);
  return true;
}

bool RectangularIntegration::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}

double RectangularIntegration::calculateIntegral(std::vector<double>& args_) {
  double integralValue = 0;
  double prevValue = 0;
  int subdivisions = 2;
  bool flag = true;

  auto [low, high] = limits.front();
  limits.erase(limits.begin());
  args_.push_back(0.0);

  while (flag) {
    prevValue = integralValue;
    integralValue = 0.0;

    double step = (high - low) / subdivisions;
    args_.back() = low + step / 2.0;

    for (int i = 0; i < subdivisions; ++i) {
      if (limits.empty()) {
        integralValue += integrandFunction(args_) * step;
      } else {
        integralValue += calculateIntegral(args_) * step;
      }
      args_.back() += step;
    }

    subdivisions *= 2;

    flag = (std::abs(integralValue - prevValue) * (1.0 / 3.0) > epsilon);
  }

  args_.pop_back();
  limits.insert(limits.begin(), {low, high});

  return integralValue;
}

}  // namespace korneeva_e_rectangular_integration_method_seq
