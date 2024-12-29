#include "mpi/naumov_b_simpson_method/include/ops_mpi.hpp"

#include <boost/mpi.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace naumov_b_simpson_method_mpi {

double integrir_1d(const func_1d_t &func, const bound_t &bound, int num_steps) {
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

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();

  double step_size = (bounds_.second - bounds_.first) / num_steps_;
  segments_.clear();

  for (int i = 0; i < num_steps_; ++i) {
    double left = bounds_.first + i * step_size;
    double right = left + step_size;
    segments_.emplace_back(left, right);
  }

  return true;
}

bool TestMPITaskSequential::validation() {
  internal_order_test();

  if (bounds_.first >= bounds_.second) {
    return false;
  }

  if (num_steps_ < 2 || num_steps_ % 2 != 0) {
    return false;
  }

  if (!function_) {
    return false;
  }

  return true;
}

bool TestMPITaskSequential::run() {
  internal_order_test();

  result_ = 0.0;
  for (const auto &segment : segments_) {
    result_ += integrir_1d(function_, segment, num_steps_);
  }

  return true;
}

bool TestMPITaskSequential::post_processing() {
  internal_order_test();

  *reinterpret_cast<double *>(taskData->outputs[0]) = result_;
  return true;
}

//-----------------------------------------------------------------------------------------------------------------------

double parallel_integrir_1d(const func_1d_t &func, double lower_bound, double upper_bound, int num_steps,
                            boost::mpi::communicator &world) {
  double step_size = (upper_bound - lower_bound) / num_steps;

  int rank = world.rank();
  int size = world.size();

  int steps_per_process = num_steps / size;
  int remaining_steps = num_steps % size;

  int extra_steps = (rank < remaining_steps) ? 1 : 0;

  double local_lower_bound =
      lower_bound + rank * steps_per_process * step_size + std::min(rank, remaining_steps) * step_size;
  double local_upper_bound = local_lower_bound + (steps_per_process + extra_steps) * step_size;

  double local_result = 0.0;

  if (steps_per_process + extra_steps > 0) {
    local_result += func(local_lower_bound);

    for (int i = 1; i < steps_per_process + extra_steps; ++i) {
      double x = local_lower_bound + i * step_size;
      int weight = (i % 2 == 0) ? 2 : 4;
      local_result += weight * func(x);
    }

    local_result += func(local_upper_bound);
    local_result *= step_size / 3.0;
  }

  double global_result = 0.0;
  boost::mpi::all_reduce(world, local_result, global_result, std::plus<>());

  return global_result;
}

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();
  int rank = world.rank();
  int size = world.size();

  int steps_per_process = num_steps_ / size;
  int remaining_steps = num_steps_ % size;

  local_steps_ = steps_per_process + (rank < remaining_steps ? 1 : 0);

  double interval_per_process = (upper_bound_ - lower_bound_) / size;
  local_lower_bound_ = lower_bound_ + rank * interval_per_process;
  local_upper_bound_ = local_lower_bound_ + interval_per_process;

  if (rank == size - 1) {
    local_upper_bound_ = upper_bound_;
  }
  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();
  if (lower_bound_ >= upper_bound_) {
    return false;
  }

  if (num_steps_ < world.size() || num_steps_ % 2 != 0) {
    return false;
  }

  if (!function_) {
    return false;
  }

  return true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();

  local_result_ = parallel_integrir_1d(function_, lower_bound_, upper_bound_, num_steps_, world);
  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();
  *reinterpret_cast<double *>(taskData->outputs[0]) = local_result_;
  return true;
}

}  // namespace naumov_b_simpson_method_mpi
