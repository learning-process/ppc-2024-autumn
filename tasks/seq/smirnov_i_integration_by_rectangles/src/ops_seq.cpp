#include "seq/smirnov_i_integration_by_rectangles/include/ops_seq.hpp"

#include <algorithm>
#include <thread>
#include <vector>
#include <exception>
#include <stdexcept>

using namespace std::chrono_literals;
bool smirnov_i_integration_by_rectangles::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  left_ = reinterpret_cast<double*>(taskData->inputs[0])[0];
  right_ = reinterpret_cast<double*>(taskData->inputs[1])[0];
  n_ = reinterpret_cast<int*>(taskData->inputs[2])[0];
  res = 0;
  return true;
}

bool smirnov_i_integration_by_rectangles::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}
bool smirnov_i_integration_by_rectangles::TestMPITaskSequential::run() {
  internal_order_test();
  res = seq_integrate_rect(f, left_, right_, n_);
  return true;
}

bool smirnov_i_integration_by_rectangles::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
void smirnov_i_integration_by_rectangles::TestMPITaskSequential::set_function(double (*func)(double)) { f = func; }

double smirnov_i_integration_by_rectangles::TestMPITaskSequential::seq_integrate_rect(double (*func)(double),
                                                                                      double left, double right,
                                                                                      int n) {
  if (func == nullptr) {
    throw std::logic_error("func is nullptr");
  }
  double res_integr = 0;
  const double self_left = left;
  const double self_right = right;
  const double len_of_rect = (self_right - self_left) / n;
  for (int i = 0; i < n; i++) {
    const double left_rect = self_left + i * len_of_rect;
    res_integr += f(left_rect + len_of_rect / 2);
  }
  res_integr *= len_of_rect;
  return res_integr;
}
