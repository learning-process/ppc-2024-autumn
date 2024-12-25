#include "seq/shulpin_i_simpson_method/include/simpson_method.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

double shulpin_simpson_method::f_x_plus_y(double x, double y) { return x + y; }
double shulpin_simpson_method::f_x_mul_y(double x, double y) { return x * y; }
double shulpin_simpson_method::f_sin_plus_cos(double x, double y) { return std::sin(x) + std::cos(y); }
double shulpin_simpson_method::f_sin_mul_cos(double x, double y) { return std::sin(x) * std::cos(y); }

inline double shulpin_simpson_method::calculate_coeff(const int* index, const int* limit) {
  if (*index == 0 || *index == *limit) {
    return 1.0;
  }
  return (*index % 2 == 0) ? 2.0 : 4.0;
}

double shulpin_simpson_method::calculate_row_sum(const int* i, const int* num_steps, const double* dx, const double* dy,
                                                 const double* a, const double* c, const func* func) {
  double row_sum = 0.0;
  double x = *a + (*i) * (*dx);
  double x_coeff = calculate_coeff(i, num_steps);

  for (int j = 0; j <= *num_steps; ++j) {
    double y = *c + j * (*dy);
    row_sum += x_coeff * calculate_coeff(&j, num_steps) * (*func)(x, y);
  }

  return row_sum;
}

double shulpin_simpson_method::seq_simpson(double a, double b, double c, double d, int N, const func& func_seq) {
  if (N % 2 != 0) {
    ++N;
  }

  double dx = (b - a) / N;
  double dy = (d - c) / N;
  double seq_sum = 0.0;

  for (int i = 0; i <= N; ++i) {
    seq_sum += calculate_row_sum(&i, &N, &dx, &dy, &a, &c, &func_seq);
  }

  return (dx * dy / 9.0) * seq_sum;
}

bool shulpin_simpson_method::SimpsonMethodSeq::pre_processing() {
  internal_order_test();

  double a_value = *reinterpret_cast<double*>(taskData->inputs[0]);
  double b_value = *reinterpret_cast<double*>(taskData->inputs[1]);
  double c_value = *reinterpret_cast<double*>(taskData->inputs[2]);
  double d_value = *reinterpret_cast<double*>(taskData->inputs[3]);
  int N_value = *reinterpret_cast<int*>(taskData->inputs[4]);

  a_seq = a_value;
  b_seq = b_value;
  c_seq = c_value;
  d_seq = d_value;
  N_seq = N_value;

  return true;
}

bool shulpin_simpson_method::SimpsonMethodSeq::validation() {
  internal_order_test();

  return ((taskData->inputs.size() == 5) && (*reinterpret_cast<int*>(taskData->inputs[4]) > 0) &&
          (*reinterpret_cast<double*>(taskData->inputs[0]) < *reinterpret_cast<double*>(taskData->inputs[1])) &&
          (*reinterpret_cast<double*>(taskData->inputs[2]) < *reinterpret_cast<double*>(taskData->inputs[3])));
}

bool shulpin_simpson_method::SimpsonMethodSeq::run() {
  internal_order_test();

  res_seq = seq_simpson(a_seq, b_seq, c_seq, d_seq, N_seq, func_seq);

  return true;
}

bool shulpin_simpson_method::SimpsonMethodSeq::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res_seq;

  return true;
}

void shulpin_simpson_method::SimpsonMethodSeq::set_seq(const func& f) { func_seq = f; }