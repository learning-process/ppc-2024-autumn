#include "seq/shulpin_monte_carlo_integration/include/monte_carlo_integral.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

double shulpin_monte_carlo_integration::fsin(double x) { return std::sin(x); }
double shulpin_monte_carlo_integration::fcos(double x) { return std::cos(x); }
double shulpin_monte_carlo_integration::f_two_sin_cos(double x) { return 2 * std::sin(x) * std::cos(x); }

double shulpin_monte_carlo_integration::integral(double a, double b, int N, const func& func_seq) {
  double h = (b - a) / (N * 1.0);
  double sum = 0.0;

  for (int i = 0; i < N; ++i) {
    sum += func_seq(a + h * i);
  }

  return h * sum;
}

bool shulpin_monte_carlo_integration::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  double a_value = *reinterpret_cast<double*>(taskData->inputs[0]);
  double b_value = *reinterpret_cast<double*>(taskData->inputs[1]);
  int N_value = *reinterpret_cast<int*>(taskData->inputs[2]);

  a_seq = a_value;
  b_seq = b_value;
  N_seq = N_value;

  return true;
}

bool shulpin_monte_carlo_integration::TestMPITaskSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == 1;
}

bool shulpin_monte_carlo_integration::TestMPITaskSequential::run() {
  internal_order_test();

  res = integral(a_seq, b_seq, N_seq, func_seq);

  return true;
}

bool shulpin_monte_carlo_integration::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;

  return true;
}

void shulpin_monte_carlo_integration::TestMPITaskSequential::set_seq(const func& f) { func_seq = f; }
