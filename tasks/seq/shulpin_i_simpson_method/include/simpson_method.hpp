#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <utility>

#include "core/task/include/task.hpp"

namespace shulpin_simpson_method {
using func = std::function<double(double, double)>;

double f_x_plus_y(double x, double y);
double f_x_mul_y(double x, double y);
double f_sin_plus_cos(double x, double y);
double f_sin_mul_cos(double x, double y);

double seq_simpson(double a, double b, double c, double d, int N, const func& f);
inline double calculate_coeff(const int* index, const int* limit);
double calculate_row_sum(const int* i, const int* num_steps, const double* dx, const double* dy, const double* a,
                         const double* c, const func* func);

class SimpsonMethodSeq : public ppc::core::Task {
 public:
  explicit SimpsonMethodSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  void set_seq(const func& f);

 private:
  double a_seq;
  double b_seq;
  double c_seq;
  double d_seq;
  int N_seq;
  func func_seq;
  double res_seq;
};
}  // namespace shulpin_simpson_method