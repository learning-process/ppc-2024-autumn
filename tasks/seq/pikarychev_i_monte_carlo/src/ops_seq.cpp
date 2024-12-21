// Copyright 2024 Nesterov Alexander
#include "seq/pikarychev_i_monte_carlo/include/ops_seq.hpp"

#include <random>
#include <thread>

using namespace std::chrono_literals;

double pikarychev_i_monte_carlo_seq::function_double(double x) { return x * x; }

bool pikarychev_i_monte_carlo_seq::TestTaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] == 4 && taskData->outputs_count[0] == 1);
}

bool pikarychev_i_monte_carlo_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  a = reinterpret_cast<double*>(taskData->inputs[0])[0];
  b = reinterpret_cast<double*>(taskData->inputs[0])[1];
  num_samples = static_cast<int>(reinterpret_cast<double*>(taskData->inputs[0])[2]);
  seed = static_cast<int>(reinterpret_cast<double*>(taskData->inputs[0])[3]);
  range_width = b - a;
  return true;
}

bool pikarychev_i_monte_carlo_seq::TestTaskSequential::run() {
  internal_order_test();
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(a, b);
  res = 0.0;
  for (int i = 0; i < num_samples; i++) {
    double x = distribution(generator);
    res += function_double(x);
  }
  res *= (b - a) / num_samples;
  return true;
}

bool pikarychev_i_monte_carlo_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
