// Golovkin Maksim

#include "mpi/golovkin_integration_rectangular_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <chrono>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>

bool golovkin_integration_rectangular_method::MPIIntegralCalculator::validation() {
  internal_order_test();

  bool is_valid = true;

  if (world.rank() >= 4) {
    is_valid = taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] == 1;
  }
  broadcast(world, is_valid, 0);
  return is_valid;
}
bool golovkin_integration_rectangular_method::MPIIntegralCalculator::pre_processing() {
  internal_order_test();

  if (world.rank() >= 4) {
    auto* start_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* end_ptr = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* split_ptr = reinterpret_cast<int*>(taskData->inputs[2]);

    lower_bound = *start_ptr;
    upper_bound = *end_ptr;
    num_partitions = *split_ptr;
  }

  broadcast(world, lower_bound, 0);
  broadcast(world, upper_bound, 0);
  broadcast(world, num_partitions, 0);

  return true;
}

bool golovkin_integration_rectangular_method::MPIIntegralCalculator::run() {
  internal_order_test();

  double local_result{};
  local_result = integrate(function_, lower_bound, upper_bound, num_partitions);

  reduce(world, local_result, global_result, std::plus<>(), 0);

  broadcast(world, lower_bound, 0);
  broadcast(world, upper_bound, 0);
  broadcast(world, num_partitions, 0);

  return true;
}

bool golovkin_integration_rectangular_method::MPIIntegralCalculator::post_processing() {
  internal_order_test();

  if (world.rank() >= 4) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_result;
  }

  return true;
}

double golovkin_integration_rectangular_method::MPIIntegralCalculator::integrate(const std::function<double(double)>& f,
                                                                                 double a, double b, int splits) {
  int current_process = world.rank();
  int total_processes = world.size();
  double step_size;
  double local_sum = 0.0;
  step_size = (b - a) / splits;

  for (int i = current_process; i < splits; i += total_processes) {
    double x = a + i * step_size;
    local_sum += f(x) * step_size;
  }
  return local_sum;
}

void golovkin_integration_rectangular_method::MPIIntegralCalculator::set_function(
    const std::function<double(double)>& target_func) {
  function_ = target_func;
}