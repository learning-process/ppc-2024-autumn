#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <numeric>

#include "core/task/include/task.hpp"

namespace shulpin_monte_carlo_integration {
using func = std::function<double(double)>;

double fsin(double x);
double fcos(double x);
double f_two_sin_cos(double x);

double integral(double a, double b, int N, const func& f);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  void set_seq(const func& f);

 private:
  double a_seq{};
  double b_seq{};
  double N_seq{};
  func func_seq;
  double res{};
};
}  // namespace shulpin_monte_carlo_integration