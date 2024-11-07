// Copyright 2024 Nesterov Alexander
#include "seq/prokhorov_n_integral_rectangle_method/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool prokhorov_n_integral_rectangle_method::TestTaskSequential::pre_processing() {
  internal_order_test();
  uint8_t* inputs_raw = taskData->inputs[0];

  std::vector<double> inputs(reinterpret_cast<double*>(inputs_raw), reinterpret_cast<double*>(inputs_raw) + 3);

  left_ = inputs[0];                 
  right_ = inputs[1];               
  n = static_cast<int>(inputs[2]);  

  res = 0.0;
  return true;
}

bool prokhorov_n_integral_rectangle_method::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 3 && taskData->outputs_count[0] == 1;
}

bool prokhorov_n_integral_rectangle_method::TestTaskSequential::run() {
  internal_order_test();
  res = integrate(func_, left_, right_, n);
  return true;
}

bool prokhorov_n_integral_rectangle_method::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

double prokhorov_n_integral_rectangle_method::TestTaskSequential::integrate(const std::function<double(double)>& f,
                                                                            double left_, double right_, int n) {
  double step = (right_ - left_) / n;
  double area = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = left_ + (i + 0.5) * step;
    area += f(x) * step;
  }

  return area;
}

void prokhorov_n_integral_rectangle_method::TestTaskSequential::set_function(
    const std::function<double(double)>& func) {
  func_ = func;
}
