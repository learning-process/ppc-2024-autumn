#include "seq/malyshev_v_monte_carlo_integration/include/malyshev_v_monte_carlo_integration.hpp"

#include <cmath>
#include <random>

using namespace malyshev_v_monte_carlo_integration;

double MonteCarloIntegration::function_to_integrate(double x) {
  return x * x;
}

bool MonteCarloIntegration::validation() {
  internal_order_test();
  return (taskData->inputs.size() == 3 && taskData->outputs.size() == 1);
}

bool MonteCarloIntegration::pre_processing() {
  internal_order_test();
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  num_points = *reinterpret_cast<int*>(taskData->inputs[2]);
  rng = std::mt19937(std::random_device{}());
  dist = std::uniform_real_distribution<double>(a, b);

  return true;
}

double MonteCarloIntegration::monte_carlo_integration() {
  double sum = 0.0;

  for (int i = 0; i < num_points; ++i) {
    double x = dist(rng);
    sum += function_to_integrate(x);
  }

  double area = (b - a) * sum / num_points;
  return area;
}

bool MonteCarloIntegration::run() {
  internal_order_test();
  result = monte_carlo_integration();
  return true;
}

bool MonteCarloIntegration::post_processing() {
  internal_order_test();
  *reinterpret_cast<double*>(taskData->outputs[0]) = result;
  return true;
}
