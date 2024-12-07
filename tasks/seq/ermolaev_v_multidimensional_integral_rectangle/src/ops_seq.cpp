// Copyright 2024 Nesterov Alexander
#include "seq/ermolaev_v_multidimensional_integral_rectangle/include/ops_seq.hpp"

#include <climits>
#include <random>

double ermolaev_v_multidimensional_integral_rectangle_seq::integrateImpl(std::deque<std::pair<double, double>>& limits,
                                                                         std::vector<double>& args,
                                                                         const function& func, double eps) {
  double I = 0;
  double I0;
  int n = 2;

  auto [a, b] = limits.front();
  limits.pop_front();
  args.push_back(double{});

  do {
    I0 = I;
    I = 0;

    double h = (b - a) / n;
    args.back() = a + h / 2;
    for (int i = 0; i < n; i++) {
      if (limits.empty())
        I += func(args) * h;
      else
        I += ermolaev_v_multidimensional_integral_rectangle_seq::integrateImpl(limits, args, func, eps) * h;

      args.back() += h;
    }

    n *= 2;
  } while (std::abs(I - I0) * 1 / 3 > eps);

  args.pop_back();
  limits.emplace_front(a, b);

  return I;
}

double ermolaev_v_multidimensional_integral_rectangle_seq::integrate(std::deque<std::pair<double, double>> limits,
                                                                     double eps, const function& func) {
  std::vector<double> args;
  return ermolaev_v_multidimensional_integral_rectangle_seq::integrateImpl(limits, args, func, eps);
}

bool ermolaev_v_multidimensional_integral_rectangle_seq::TestTaskSequential::pre_processing() {
  internal_order_test();

  auto* ptr = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
  limits_.assign(ptr, ptr + taskData->inputs_count[0]);
  eps_ = *reinterpret_cast<double*>(taskData->inputs[1]);

  return true;
}

bool ermolaev_v_multidimensional_integral_rectangle_seq::TestTaskSequential::validation() {
  internal_order_test();

  return taskData->inputs_count[0] > 0 && taskData->inputs.size() == 2 && taskData->outputs_count[0] == 1 &&
         !taskData->outputs.empty();
}

bool ermolaev_v_multidimensional_integral_rectangle_seq::TestTaskSequential::run() {
  internal_order_test();

  res_ = integrate(limits_, eps_, function_);

  return true;
}

bool ermolaev_v_multidimensional_integral_rectangle_seq::TestTaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = res_;

  return true;
}
