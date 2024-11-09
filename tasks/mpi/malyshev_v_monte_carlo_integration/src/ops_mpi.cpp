#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <random>

namespace malyshev_v_monte_carlo_integration {

bool TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs.size() == 3 && taskData->outputs.size() == 1);
}

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  double input_epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
  epsilon = input_epsilon;
  num_samples = static_cast<int>((b - a) * 100 / epsilon);
  if (num_samples < 10) {
    num_samples = 10;
  }
  return true;
}

bool TestMPITaskSequential::run() {
  internal_order_test();
  double h = (b - a) / num_samples;
  double sum = (function(a) + function(b)) / 2.0;

  for (int i = 1; i < num_samples; ++i) {
    sum += function(a + i * h);
  }

  res = h * sum;
  return true;
}

bool TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if ((taskData->inputs.size() != 3) || (taskData->outputs.size() != 1)) {
      return false;
    }
    double input_epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
    if (input_epsilon <= 0) {
      return false;
    }
  }
  return true;
}

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    a = *reinterpret_cast<double*>(taskData->inputs[0]);
    b = *reinterpret_cast<double*>(taskData->inputs[1]);
    double input_epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
    epsilon = input_epsilon;
    num_samples = static_cast<int>((b - a) * 100 / epsilon);
    if (num_samples < 10) {
      num_samples = 10;
    }
  }

  boost::mpi::broadcast(world, a, 0);
  boost::mpi::broadcast(world, b, 0);
  boost::mpi::broadcast(world, num_samples, 0);
  local_num_samples = num_samples / world.size();
  return true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();
  double h = (b - a) / num_samples;
  double local_sum = 0.0;

  for (int i = world.rank() * local_num_samples; i < (world.rank() + 1) * local_num_samples; ++i) {
    double x = a + i * h;
    local_sum += function(x);
  }

  if (world.rank() == 0) {
    local_sum += (function(a) + function(b)) / 2.0;
  }

  local_sum *= h;
  boost::mpi::reduce(world, local_sum, res, std::plus<>(), 0);
  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  }
  return true;
}

}  // namespace malyshev_v_monte_carlo_integration