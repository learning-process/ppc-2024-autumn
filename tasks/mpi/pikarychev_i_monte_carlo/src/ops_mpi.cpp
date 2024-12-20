// Copyright 2023 Nesterov Alexander
#include "mpi/pikarychev_i_monte_carlo/include/ops_mpi.hpp"

#include <iostream>
#include <random>

std::vector<int> pikarychev_i_monte_carlo_parallel::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool pikarychev_i_monte_carlo_parallel::TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] == 4 && taskData->outputs_count[0] == 1);
}

bool pikarychev_i_monte_carlo_parallel::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  a = reinterpret_cast<double*>(taskData->inputs[0])[0];
  b = reinterpret_cast<double*>(taskData->inputs[0])[1];
  num_samples = static_cast<int>(reinterpret_cast<double*>(taskData->inputs[0])[2]);
  seed = static_cast<int>(reinterpret_cast<double*>(taskData->inputs[0])[3]);

  range_width = b - a;
  return true;
}

bool pikarychev_i_monte_carlo_parallel::TestMPITaskSequential::run() {
  internal_order_test();
  a = reinterpret_cast<double*>(taskData->inputs[0])[0];
  b = reinterpret_cast<double*>(taskData->inputs[0])[1];
  num_samples = static_cast<int>(reinterpret_cast<double*>(taskData->inputs[0])[2]);
  seed = static_cast<int>(reinterpret_cast<double*>(taskData->inputs[0])[3]);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(a, b);

  for (int i = 0; i < num_samples; i++) {
    double x = distribution(generator);
    res += function_double(x);
  }
  res *= (b - a) / num_samples;

  return true;
}

bool pikarychev_i_monte_carlo_parallel::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

double pikarychev_i_monte_carlo_parallel::function_double(double x) { return x * x; }

bool pikarychev_i_monte_carlo_parallel::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) return (taskData->inputs_count[0] == 4 && taskData->outputs_count[0] == 1);
  return true;
}

bool pikarychev_i_monte_carlo_parallel::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    a = reinterpret_cast<double*>(taskData->inputs[0])[0];
    b = reinterpret_cast<double*>(taskData->inputs[0])[1];
    num_samples = (int)(reinterpret_cast<double*>(taskData->inputs[0])[2]);
    seed = (int)(reinterpret_cast<double*>(taskData->inputs[0])[3]);
    range_width = b - a;
  }
  return true;
}

bool pikarychev_i_monte_carlo_parallel::TestMPITaskParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, a, 0);
  boost::mpi::broadcast(world, b, 0);
  boost::mpi::broadcast(world, num_samples, 0);
  boost::mpi::broadcast(world, seed, 0);
  boost::mpi::broadcast(world, range_width, 0);

  const int rank = world.rank();
  const int size = world.size();
  int local_samples = num_samples / size;
  if (rank < num_samples % size) local_samples++;

  std::mt19937 generator(seed + rank);
  std::uniform_real_distribution<double> distribution(a, b);

  double local_sum = 0.0;
  double x;
  for (int i = 0; i < local_samples; i++) {
    x = distribution(generator);
    local_sum += function_double(x);
  }
  std::cout << "ls=" << local_sum << std::endl;
  double global_sum = 0.0;
  boost::mpi::reduce(world, local_sum, global_sum, std::plus<>(), 0);

  if (world.rank() == 0) {
    res = (double)(global_sum * (range_width / num_samples));
  }
  return true;
}

bool pikarychev_i_monte_carlo_parallel::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  }
  return true;
}