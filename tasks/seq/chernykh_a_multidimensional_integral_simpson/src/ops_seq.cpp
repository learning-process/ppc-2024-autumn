#include "seq/chernykh_a_multidimensional_integral_simpson/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace chernykh_a_multidimensional_integral_simpson_seq {

double integrate_1d(const std::function<double(double)> &func,  //
                    const std::pair<double, double> &bound,     //
                    int num_steps) {
  auto [lower_bound, upper_bound] = bound;
  double step_size = (upper_bound - lower_bound) / num_steps;
  double result = func(lower_bound) + func(upper_bound);

  for (int step_index = 1; step_index < num_steps; ++step_index) {
    double func_arg = lower_bound + (step_index * step_size);
    int weight = (step_index % 2 == 0) ? 2 : 4;
    result += weight * func(func_arg);
  }

  return result * step_size / 3.0;
}

double integrate_nd(const std::function<double(const std::vector<double> &)> &func,  //
                    std::vector<double> &func_args,                                  //
                    const std::vector<std::pair<double, double>> &bounds,            //
                    const std::pair<int, int> &step_range,                           //
                    double tolerance,                                                //
                    int dim) {
  auto [min_steps, max_steps] = step_range;
  double prev_steps_result = 0.0;
  double curr_steps_result = 0.0;

  for (int num_steps = min_steps; num_steps <= max_steps; num_steps *= 2) {
    prev_steps_result = curr_steps_result;
    curr_steps_result = integrate_1d(
        [&](double func_arg) -> double {
          func_args[dim] = func_arg;
          return (dim == 0) ? func(func_args) : integrate_nd(func, func_args, bounds, step_range, tolerance, dim - 1);
        },
        bounds[dim], num_steps);

    if (std::isnan(curr_steps_result)) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    if (std::abs(curr_steps_result - prev_steps_result) < tolerance) {
      return curr_steps_result;
    }
  }

  return std::numeric_limits<double>::quiet_NaN();
}

bool SequentialTask::validation() {
  internal_order_test();
  auto *bounds_ptr = reinterpret_cast<std::pair<double, double> *>(taskData->inputs[1]);
  auto bounds_size = taskData->inputs_count[1];
  auto *step_range_ptr = reinterpret_cast<std::pair<int, int> *>(taskData->inputs[2]);
  auto *tolerance_ptr = reinterpret_cast<double *>(taskData->inputs[3]);

  auto is_valid_bounds = std::all_of(bounds_ptr, bounds_ptr + bounds_size, [](auto &b) { return b.first < b.second; });

  return bounds_size > 0 &&                                 // At least one dimension is required for integration
         is_valid_bounds &&                                 // Bounds for each dimension must be correct (lower < upper)
         step_range_ptr->first < step_range_ptr->second &&  // Minimum step count must be less than maximum step count
         step_range_ptr->first % 2 == 0 &&  // Minimum step count must be even (Simpson's rule requirement)
         step_range_ptr->first >= 2 &&      // Minimum step count must be at least 2
         *tolerance_ptr > 0;                // Tolerance must be positive
}

bool SequentialTask::pre_processing() {
  internal_order_test();
  func = *reinterpret_cast<std::function<double(const std::vector<double> &)> *>(taskData->inputs[0]);
  auto *bounds_ptr = reinterpret_cast<std::pair<double, double> *>(taskData->inputs[1]);
  auto bounds_size = taskData->inputs_count[1];
  bounds.assign(bounds_ptr, bounds_ptr + bounds_size);
  step_range = *reinterpret_cast<std::pair<int, int> *>(taskData->inputs[2]);
  tolerance = *reinterpret_cast<double *>(taskData->inputs[3]);

  func_args.resize(bounds_size, 0.0);
  return true;
}

bool SequentialTask::run() {
  internal_order_test();
  result = integrate_nd(func, func_args, bounds, step_range, tolerance, int(bounds.size()) - 1);
  return true;
}

bool SequentialTask::post_processing() {
  internal_order_test();
  *reinterpret_cast<double *>(taskData->outputs[0]) = result;
  return true;
}

}  // namespace chernykh_a_multidimensional_integral_simpson_seq
