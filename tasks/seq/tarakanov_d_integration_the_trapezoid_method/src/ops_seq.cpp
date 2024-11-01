// Copyright 2024 Tarakanov Denis
#include "seq/tarakanov_d_integration_the_trapezoid_method/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

bool tarakanov_d_integration_the_trapezoid_method_seq::integration_the_trapezoid_method::pre_processing() {
  internal_order_test();
  // Init value for input and output
  a = reinterpret_cast<double*>(taskData->inputs[0])[0];
  b = reinterpret_cast<double*>(taskData->inputs[0])[1];
  h = reinterpret_cast<double*>(taskData->inputs[0])[2];
  res = 0;
  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_seq::integration_the_trapezoid_method::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 4 && taskData->outputs_count[0] == 1;
}

bool tarakanov_d_integration_the_trapezoid_method_seq::integration_the_trapezoid_method::run() {
  internal_order_test();
  double integral = 0.5 * (f(a) + f(b));
  for (double x = a + h; x < b; x += h) {
    integral += f(x);
  }
  res = static_cast<int>(integral * h);
  return true;
}

bool tarakanov_d_integration_the_trapezoid_method_seq::integration_the_trapezoid_method::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  return true;
}
