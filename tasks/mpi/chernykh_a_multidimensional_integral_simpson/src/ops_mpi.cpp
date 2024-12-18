#include "mpi/chernykh_a_multidimensional_integral_simpson/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace chernykh_a_multidimensional_integral_simpson_mpi {

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

bool ParallelTask::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    auto *bounds_ptr = reinterpret_cast<std::pair<double, double> *>(taskData->inputs[0]);
    auto bounds_size = taskData->inputs_count[0];
    auto *step_range_ptr = reinterpret_cast<std::pair<int, int> *>(taskData->inputs[1]);
    auto *tolerance_ptr = reinterpret_cast<double *>(taskData->inputs[2]);

    auto is_valid_bounds =
        std::all_of(bounds_ptr, bounds_ptr + bounds_size, [](auto &b) { return b.first < b.second; });

    return bounds_size > 0 &&                                 // At least one dimension is required for integration
           is_valid_bounds &&                                 // Bounds for each dimension must be valid (lower < upper)
           step_range_ptr->first < step_range_ptr->second &&  // Minimum step count must be less than maximum step count
           step_range_ptr->first % 2 == 0 &&  // Minimum step count must be even (Simpson's rule requirement)
           step_range_ptr->first >= 2 &&      // Minimum step count must be at least 2
           *tolerance_ptr > 0;                // Tolerance must be positive
  }
  return true;
}

bool ParallelTask::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto *bounds_ptr = reinterpret_cast<std::pair<double, double> *>(taskData->inputs[0]);
    auto bounds_size = taskData->inputs_count[0];
    bounds.assign(bounds_ptr, bounds_ptr + bounds_size);
    step_range = *reinterpret_cast<std::pair<int, int> *>(taskData->inputs[1]);
    tolerance = *reinterpret_cast<double *>(taskData->inputs[2]);

    func_args.resize(bounds.size(), 0.0);
  }
  return true;
}

bool ParallelTask::run() {
  internal_order_test();
  boost::mpi::broadcast(world, bounds, 0);
  boost::mpi::broadcast(world, step_range, 0);
  boost::mpi::broadcast(world, tolerance, 0);
  boost::mpi::broadcast(world, func_args, 0);
  int dim = int(bounds.size()) - 1;

  auto [lower_bound, upper_bound] = bounds[dim];
  double chunk_size = (upper_bound - lower_bound) / world.size();
  std::pair<double, double> chunk_bound = {
      lower_bound + (chunk_size * world.rank()),
      lower_bound + (chunk_size * (world.rank() + 1)),
  };

  auto [min_steps, max_steps] = step_range;
  bool is_converged = false;
  double curr_steps_result = 0.0;
  double prev_steps_result = 0.0;

  for (int num_steps = min_steps; !is_converged; num_steps *= 2) {
    prev_steps_result = curr_steps_result;
    double chunk_result = integrate_1d(
        [&](double func_arg) -> double {
          func_args[dim] = func_arg;
          return (dim == 0) ? func(func_args) : integrate_nd(func, func_args, bounds, step_range, tolerance, dim - 1);
        },
        chunk_bound, num_steps);
    boost::mpi::reduce(world, chunk_result, curr_steps_result, std::plus(), 0);

    if (world.rank() == 0) {
      if (std::isnan(curr_steps_result) || num_steps > max_steps) {
        result = std::numeric_limits<double>::quiet_NaN();
        is_converged = true;
      }
      if (std::abs(curr_steps_result - prev_steps_result) < tolerance) {
        result = curr_steps_result;
        is_converged = true;
      }
    }
    boost::mpi::broadcast(world, is_converged, 0);
  }

  return true;
}

bool ParallelTask::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double *>(taskData->outputs[0]) = result;
  }
  return true;
}

}  // namespace chernykh_a_multidimensional_integral_simpson_mpi
