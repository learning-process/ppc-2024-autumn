#include "seq/plekhanov_d_trapez_integration/include/ops_seq.hpp"

#include <functional>

bool plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential::pre_processing() {
  internal_order_test();
  auto* inputs = reinterpret_cast<double*>(taskData->inputs[0]);
  a_ = inputs[0];
  b_ = inputs[1];
  n_ = static_cast<int>(inputs[2]);
  res_ = 0.0;
  return true;
}

bool plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 3 && taskData->outputs_count[0] == 1;
}

bool plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential::run() {
  internal_order_test();
  res_ = integrate_function(a_, b_, n_, function_);
  return true;
}

bool plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res_;
  return true;
}

void plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential::set_function(std::function<double(double)> f) {
  function_ = std::move(f);
}

double plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential::integrate_function(
    double a, double b, int n, const std::function<double(double)>& f) {
  const double width = (b - a) / n;

  double result = 0.0;
  for (int step = 0; step < n; step++) {
    const double x1 = a + step * width;
    const double x2 = a + (step + 1) * width;

    result += 0.5 * (x2 - x1) * (f(x1) + f(x2));
  }

  return result;
}