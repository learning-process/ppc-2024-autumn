#pragma once

#include <cmath>
#include <functional>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"

namespace korneeva_e_rectangular_integration_method_seq {

// Type definition for the integrand function
using Function = std::function<double(const std::vector<double>& args)>;

class RectangularIntegration : public ppc::core::Task {
 public:
  explicit RectangularIntegration(const std::shared_ptr<ppc::core::TaskData>& taskData_, Function func)
      : Task(std::move(taskData_)), integrandFunction(std::move(func)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  Function integrandFunction;                     // Integrand function
  std::vector<std::pair<double, double>> limits;  // Integration bounds
  double result;                                  // Computed integral result
  double epsilon;                                 // Precision (replacing step size)

  static constexpr double MIN_EPSILON = 1e-6;  // Minimum allowable precision
};

bool RectangularIntegration::pre_processing() {
  internal_order_test();

  // Extract integration bounds and store them in the limits vector
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

  // Validate input parameters and output
  bool validInput = taskData->inputs_count[0] > 0 && taskData->inputs.size() == 2;
  bool validOutput = taskData->outputs_count[0] == 1 && !taskData->outputs.empty();

  // Validate limits: lower bound must not be greater than upper bound
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

  size_t numDimensions = limits.size();
  std::vector<double> args(numDimensions, 0.0);  // Current arguments for integration
  double integral = 0.0;                         // Current value of the integral
  double prevIntegral;                           // Previous value of the integral for convergence check
  size_t divisions = 1;                          // Initial number of divisions

  do {
    prevIntegral = integral;
    integral = 0.0;

    // Compute step size for each dimension
    std::vector<double> stepSizes(numDimensions);
    for (size_t i = 0; i < numDimensions; ++i) {
      stepSizes[i] = (limits[i].second - limits[i].first) / divisions;
    }

    // Iterate over all combinations of grid nodes
    std::vector<size_t> indices(numDimensions, 0);
    while (true) {
      // Calculate the current point
      for (size_t i = 0; i < numDimensions; ++i) {
        args[i] = limits[i].first + stepSizes[i] * (indices[i] + 0.5);
      }

      // Add the function value at the current point to the integral
      double term = integrandFunction(args);
      for (size_t i = 0; i < numDimensions; ++i) {
        term *= stepSizes[i];
      }
      integral += term;

      // Update indices
      bool done = true;
      for (size_t i = 0; i < numDimensions; ++i) {
        if (++indices[i] < divisions) {
          done = false;
          break;
        }
        indices[i] = 0;
      }
      if (done) break;
    }

    // Double the number of divisions for the next iteration
    divisions *= 2;
  } while (std::abs(integral - prevIntegral) > epsilon);

  result = integral;
  return true;
}

bool RectangularIntegration::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}
}  // namespace korneeva_e_rectangular_integration_method_seq
