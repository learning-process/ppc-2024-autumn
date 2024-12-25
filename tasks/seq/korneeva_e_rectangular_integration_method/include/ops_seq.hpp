#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
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

  // Method to calculate the integral using the rectangular method with specified precision
  double calculateIntegral();

 private:
  Function integrandFunction;                     // Integrand function
  std::vector<std::pair<double, double>> limits;  // Integration bounds
  double result;                                  // Computed integral result
  double epsilon;                                 // Precision (replacing step size)

  static constexpr double DEFAULT_EPSILON = 1e-4;  // Default precision
  static constexpr double MIN_EPSILON = 1e-6;      // Minimum allowable precision
};

bool RectangularIntegration::pre_processing() {
  internal_order_test();

  // Extract integration bounds and store them in the limits vector
  auto* ptrInput = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
  size_t numDimensions = taskData->inputs_count[0];
  limits.assign(ptrInput, ptrInput + numDimensions);

  // Extract precision if provided; use default value otherwise
  if (taskData->inputs_count.size() > 1 && taskData->inputs_count[1] > 0) {
    epsilon = *reinterpret_cast<double*>(taskData->inputs[1]);
  } else {
    epsilon = DEFAULT_EPSILON;  // Default precision
  }

  // Validate precision
  if (epsilon < MIN_EPSILON) {
    epsilon = MIN_EPSILON;  // Enforce minimum precision
  }

  result = 0.0;  // Initialize the result
  return true;
}

bool RectangularIntegration::validation() {
  internal_order_test();

  size_t numDimensions = taskData->inputs_count[0];

  // Validate input count matches the number of dimensions
  if (taskData->inputs_count[0] <= 0) {
    return false;
  }

  // Validate there is exactly one output
  if (taskData->outputs_count[0] != 1) {
    return false;
  }

  // Validate integration bounds
  auto* ptrInput = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
  for (size_t i = 0; i < numDimensions; ++i) {
    if (ptrInput[i].first > ptrInput[i].second) {
      return false;  // Minimum bound must not exceed maximum bound
    }
  }

  // Validate precision is positive
  if (epsilon <= 0) {
    return false;
  }

  return true;
}

bool RectangularIntegration::run() {
  internal_order_test();
  result = calculateIntegral();  // Perform the integration
  return true;
}

bool RectangularIntegration::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}

double RectangularIntegration::calculateIntegral() {
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
        } else {
          indices[i] = 0;
        }
      }
      if (done) break;
    }

    // Double the number of divisions for the next iteration
    divisions *= 2;
  } while (std::abs(integral - prevIntegral) > epsilon);

  return integral;
}

}  // namespace korneeva_e_rectangular_integration_method_seq
