#include "mpi/malyshev_v_monte_carlo_integration/include/malyshev_v_monte_carlo_integration.hpp"

#include <algorithm>
#include <random>

bool malyshev_v_monte_carlo_integration::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs.size() == 3 && taskData->outputs.size() == 1);
}

bool malyshev_v_monte_carlo_integration::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  n_samples = *reinterpret_cast<int*>(taskData->inputs[2]);
  return true;
}

bool malyshev_v_monte_carlo_integration::TestMPITaskSequential::run() {
  internal_order_test();
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(a, b);

  double sum = 0.0;
  for (int i = 0; i < n_samples; ++i) {
    double x = distribution(generator);
    sum += function_square(x);
  }

  res = (b - a) * sum / n_samples;
  return true;
}

bool malyshev_v_monte_carlo_integration::TestMPITaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  return true;
}

bool malyshev_v_monte_carlo_integration::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs.size() == 3 && taskData->outputs.size() == 1);
  }
  return true;
}

bool malyshev_v_monte_carlo_integration::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    a = *reinterpret_cast<double*>(taskData->inputs[0]);
    b = *reinterpret_cast<double*>(taskData->inputs[1]);
    n_samples = *reinterpret_cast<int*>(taskData->inputs[2]);
  }

  boost::mpi::broadcast(world, a, 0);
  boost::mpi::broadcast(world, b, 0);
  boost::mpi::broadcast(world, n_samples, 0);
  return true;
}

bool malyshev_v_monte_carlo_integration::TestMPITaskParallel::run() {
  internal_order_test();
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(a, b);

  int local_n_samples = n_samples / world.size();
  if (world.rank() < n_samples % world.size()) {
    local_n_samples++;
  }

  double local_sum = 0.0;
  for (int i = 0; i < local_n_samples; ++i) {
    double x = distribution(generator);
    local_sum += function_square(x);
  }

  local_res = (b - a) * local_sum / local_n_samples;
  boost::mpi::reduce(world, local_res, res, std::plus<>(), 0);
  return true;
}

bool malyshev_v_monte_carlo_integration::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  }
  return true;
}
