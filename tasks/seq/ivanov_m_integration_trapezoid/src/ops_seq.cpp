// Copyright 2024 Ivanov Mike
#include "seq/ivanov_m_integration_trapezoid/include/ops_seq.hpp"

bool ivanov_m_integration_trapezoid_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 3 && taskData->outputs_count[0] == 1;
}

bool ivanov_m_integration_trapezoid_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  auto* input = reinterpret_cast<double*>(taskData->inputs[0]);
  a_ = input[0];
  b_ = input[1];
  n_ = static_cast<int>(input[2]);
  result_ = 0.0;

  return true;
}

bool ivanov_m_integration_trapezoid_seq::TestTaskSequential::run() {
  internal_order_test();
  if (a_ == b_) return true;
  double step_ = (b_ - a_) / n_;
  for (int i = 0; i < n_; i++) result_ += (f_(a_ + i * step_) + f_(a_ + (i + 1) * step_));
  result_ = result_ / 2 * step_;
  return true;
}

bool ivanov_m_integration_trapezoid_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  return true;
}

void ivanov_m_integration_trapezoid_seq::TestTaskSequential::add_function(const std::function<double(double)>& f) {
  f_ = f;
}