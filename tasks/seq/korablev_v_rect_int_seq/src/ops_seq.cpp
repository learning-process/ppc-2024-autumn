#include "seq/korablev_v_rect_int_seq/include/ops_seq.hpp"

#include <functional>
#include <string>
#include <thread>

using namespace std::chrono_literals;

bool korablev_v_rect_int_seq::RectangularIntegrationSequential::pre_processing() {
  internal_order_test();

  auto* inputs = reinterpret_cast<double*>(taskData->inputs[0]);

  a_ = inputs[0];
  b_ = inputs[1];
  n_ = static_cast<int>(inputs[2]);

  result_ = 0.0;
  return true;
}

bool korablev_v_rect_int_seq::RectangularIntegrationSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 3 && taskData->outputs_count[0] == 1;
}

bool korablev_v_rect_int_seq::RectangularIntegrationSequential::run() {
  internal_order_test();

  result_ = integrate(func_, a_, b_, n_);

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  return true;
}

bool korablev_v_rect_int_seq::RectangularIntegrationSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  return true;
}

double korablev_v_rect_int_seq::RectangularIntegrationSequential::integrate(const std::function<double(double)>& f,
                                                                            double a, double b, int n) {
  double step = (b - a) / n;
  double area = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = a + (i + 0.5) * step;
    area += f(x) * step;
  }

  return area;
}

void korablev_v_rect_int_seq::RectangularIntegrationSequential::set_function(
    const std::function<double(double)>& func) {
  func_ = func;
}