// Golovkin Maksim Task#1
#include "mpi/golovkin_integration_rectangular_method/include/ops_mpi.hpp"

#include <boost/mpi.hpp>

using namespace golovkin_integration_rectangular_method;

bool MPIIntegralCalculator::validation() {
  internal_order_test();

  bool is_valid = true;

  if (world.size() < 5 || world.rank() >= 4) {
    is_valid = taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] == 1;

    if (!is_valid) {
      std::cerr << "Validation failed on rank 0 with inputs_count or outputs_count invalid\n";
    }
  }

  broadcast(world, is_valid, 0);

  return is_valid;
}
bool MPIIntegralCalculator::pre_processing() {
  internal_order_test();

  if (world.size() < 5 || world.rank() >= 4) {
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

bool MPIIntegralCalculator::run() {
  internal_order_test();

  if (world.size() < 5 || world.rank() >= 4) {
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

  double local_result = integrate(function_, lower_bound, upper_bound, num_partitions);

  reduce(world, local_result, global_result, std::plus<>(), 0);

  if (world.size() < 5 || world.rank() >= 4) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_result;
  }

  return true;
}

bool MPIIntegralCalculator::post_processing() {
  internal_order_test();

  if (world.size() < 5 || world.rank() >= 4) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_result;
  }

  return true;
}

double MPIIntegralCalculator::integrate(const std::function<double(double)>& f, double a, double b, int splits) {
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

void MPIIntegralCalculator::set_function(const std::function<double(double)>& target_func) { function_ = target_func; }