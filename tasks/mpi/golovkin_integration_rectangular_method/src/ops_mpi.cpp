// Golovkin Maksim
#include "mpi/golovkin_integration_rectangular_method/include/ops_mpi.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <boost/mpi.hpp>

using namespace golovkin_integration_rectangular_method;
using namespace std::chrono_literals;
bool MPIIntegralCalculator::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool MPIIntegralCalculator::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
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
double MPIIntegralCalculator::integrate(const std::function<double(double)>& f, double a, double b, int splits) {
  int current_process = world.rank();
  int total_processes = world.size();
  double step_size;
  double local_sum = 0.0;
  step_size = (b - a) / splits;  // Вычисление ширины подынтервала
  for (int i = current_process; i < splits; i += total_processes) {
    double x = a + i * step_size;
    local_sum += f(x) * step_size;
  }
  return local_sum;
}
bool MPIIntegralCalculator::run() {
  internal_order_test();
  double local_result{};
  local_result = integrate(function_, lower_bound, upper_bound, num_partitions);
  reduce(world, local_result, global_result, std::plus<>(), 0);
  return true;
}

bool MPIIntegralCalculator::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_result;
  }
  return true;
}

void MPIIntegralCalculator::set_function(const std::function<double(double)>& target_func) {
  function_ = target_func;
}