#include "seq/shuravina_o_monte_carlo/include/ops_seq.hpp"

#include <iostream>
#include <random>

bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskSequential::pre_processing() {
  internal_order_test();
  integral_value_ = 0.0;
  return true;
}

bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs_count.size() != 1 || taskData->outputs_count.size() != 1) {
    return false;
  }
  if (taskData->inputs_count[0] != 0 || taskData->outputs_count[0] != 1) {
    return false;
  }
  if (taskData->inputs[0] != nullptr || taskData->outputs[0] == nullptr) {
    return false;
  }
  return true;
}

bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskSequential::run() {
  internal_order_test();
  auto f = [](double x) { return x * x; };
  double a = 0.0;
  double b = 1.0;
  int num_points = 1000000;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(a, b);

  double sum = 0.0;
  for (int i = 0; i < num_points; ++i) {
    double x = dis(gen);
    sum += f(x);
  }

  integral_value_ = (sum / num_points) * (b - a);

  return true;
}

bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = integral_value_;
  return true;
}