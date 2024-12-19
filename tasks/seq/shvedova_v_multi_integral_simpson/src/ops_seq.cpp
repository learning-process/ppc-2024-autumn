#include "seq/shvedova_v_multi_integral_simpson/include/ops_seq.hpp"

#include <cmath>
#include <vector>

namespace shvedova_v_multidimensional_integral_simpson_seq {

double simpsonIntegrateImpl(limits& integrationLimits, std::vector<double>& currentArgs,
                            const FunctionType& targetFunction, double tolerance) {
  int subintervalCount = 2;
  double previousIntegralValue;
  double integralValue = 0.0;

  auto [lowerLimit, upperLimit] = integrationLimits.front();
  integrationLimits.pop_front();
  currentArgs.push_back(double{});

  do {
    previousIntegralValue = integralValue;
    integralValue = 0.0;

    double stepSize = (upperLimit - lowerLimit) / subintervalCount;
    currentArgs.back() = lowerLimit;

    if (integrationLimits.empty())
      integralValue += targetFunction(currentArgs);
    else
      integralValue += simpsonIntegrateImpl(integrationLimits, currentArgs, targetFunction, tolerance);

    for (int i = 1; i < subintervalCount; i++) {
      currentArgs.back() = lowerLimit + i * stepSize;
      double weight = (i % 2 == 0) ? 2 : 4;

      if (integrationLimits.empty())
        integralValue += weight * targetFunction(currentArgs);
      else
        integralValue += weight * simpsonIntegrateImpl(integrationLimits, currentArgs, targetFunction, tolerance);
    }
    currentArgs.back() = upperLimit;
    if (integrationLimits.empty())
      integralValue += targetFunction(currentArgs);
    else
      integralValue += simpsonIntegrateImpl(integrationLimits, currentArgs, targetFunction, tolerance);

    integralValue *= stepSize / 3;
    subintervalCount *= 2;

  } while (std::abs(integralValue - previousIntegralValue) > tolerance);

  currentArgs.pop_back();
  integrationLimits.emplace_front(lowerLimit, upperLimit);

  return integralValue;
}

double simpsonIntegrate(limits integrationLimits, double tolerance, const FunctionType& targetFunction) {
  std::vector<double> currentArgs;
  return simpsonIntegrateImpl(integrationLimits, currentArgs, targetFunction, tolerance);
}

bool MultidimensionalIntegralSequential::pre_processing() {
  internal_order_test();

  auto* inputLimits = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
  integrationLimits_.assign(inputLimits, inputLimits + taskData->inputs_count[0]);
  targetFunction_ = *reinterpret_cast<FunctionType*>(taskData->inputs[1]);
  tolerance_ = *reinterpret_cast<double*>(taskData->inputs[2]);

  return true;
}

bool MultidimensionalIntegralSequential::validation() {
  internal_order_test();
  return taskData->inputs.size() == 3 && taskData->inputs_count[0] > 0 && !taskData->outputs.empty() &&
         taskData->outputs_count[0] == 1;
}

bool MultidimensionalIntegralSequential::run() {
  internal_order_test();
  result_ = simpsonIntegrate(integrationLimits_, tolerance_, targetFunction_);
  return true;
}

bool MultidimensionalIntegralSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  return true;
}

}  // namespace shvedova_v_multidimensional_integral_simpson_seq