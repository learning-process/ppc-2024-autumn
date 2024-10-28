#include "seq/gusev_n_trapezoidal_rule/include/ops_seq.hpp"

#include <functional>
#include <string>
#include <thread>

using namespace std::chrono_literals;

bool gusev_n_trapezoidal_rule_seq::TrapezoidalIntegrationSequential::pre_processing() {
  internal_order_test();

  auto* inputs = reinterpret_cast<double*>(taskData->inputs[0]);

  a_ = inputs[0];
  b_ = inputs[1];
  n_ = static_cast<int>(inputs[2]);

  result_ = 0.0;
  return true;
}

bool gusev_n_trapezoidal_rule_seq::TrapezoidalIntegrationSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 3 && taskData->outputs_count[0] == 1;
}

bool gusev_n_trapezoidal_rule_seq::TrapezoidalIntegrationSequential::run() {
  internal_order_test();

  result_ = integrate(func_, a_, b_, n_);

  return true;
}

bool gusev_n_trapezoidal_rule_seq::TrapezoidalIntegrationSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  return true;
}

double gusev_n_trapezoidal_rule_seq::TrapezoidalIntegrationSequential::integrate(const std::function<double(double)>& f,
                                                                                 double a, double b, int n) {
  double step = (b - a) / n;
  double area = 0.0;

  for (int i = 0; i < n; ++i) {
    double x0 = a + i * step;
    double x1 = a + (i + 1) * step;
    area += (f(x0) + f(x1)) * step / 2.0;
  }

  return area;
}

void gusev_n_trapezoidal_rule_seq::TrapezoidalIntegrationSequential::set_function(
    const std::function<double(double)>& func) {
  func_ = func;
}