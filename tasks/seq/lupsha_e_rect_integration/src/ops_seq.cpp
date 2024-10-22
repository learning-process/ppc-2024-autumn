// Copyright 2024 Lupsha Egor
#include "seq/lupsha_e_rect_integration/include/ops_seq.hpp"

#include <algorithm>

using namespace std::chrono_literals;

void lupsha_e_rect_integration_seq::TestTaskSequential::function_set(const std::function<double(double)>& func) {
  f = func;
}

bool lupsha_e_rect_integration_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  lower_bound = *reinterpret_cast<double*>(taskData->inputs[0]);
  upper_bound = *reinterpret_cast<double*>(taskData->inputs[1]);
  num_intervals = *reinterpret_cast<int*>(taskData->inputs[2]);

  results_.resize(1, 0.0);
  return true;
}

bool lupsha_e_rect_integration_seq::TestTaskSequential::validation() {
  internal_order_test();
  if (taskData->inputs.size() < 3) {
    return false;
  }
  return true;
}

bool lupsha_e_rect_integration_seq::TestTaskSequential::run() {
  internal_order_test();
  double width = (upper_bound - lower_bound) / num_intervals;
  input_.resize(num_intervals);
  double sum = 0.0;

  for (int i = 0; i < num_intervals; ++i) {
    double x = lower_bound + i * width;
    sum += f(x) * width;
  }
  results_[0] = sum;
  return true;
}

bool lupsha_e_rect_integration_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<double*>(taskData->outputs[0]) = results_[0];
  return true;
}
  // namespace lupsha_e_rect_integration_seq