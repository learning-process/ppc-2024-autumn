// Copyright 2024 Nesterov Alexander

#include "seq/fyodorov_m_trapezoidal_method_seq/include/ops_seq.hpp"

#include <iostream>
#include <numeric>
#include <thread>

using namespace std::chrono_literals;

double fyodorov_m_trapezoidal_method_seq::TestTaskSequential::evaluate_func(
    const std::function<double(const std::vector<double>&)>& func, const std::vector<double>& point) {
  return func(point);
}

bool fyodorov_m_trapezoidal_method_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  if (taskData->inputs_count.size() < 4) return false;  //

  func_ = *reinterpret_cast<std::function<double(const std::vector<double>&)>*>(taskData->inputs[0]);

  size_t dim = taskData->inputs_count[1];
  lower_bounds_.resize(dim);
  upper_bounds_.resize(dim);
  intervals_.resize(dim);

  for (size_t i = 0; i < dim; ++i) {
    lower_bounds_[i] = reinterpret_cast<double*>(taskData->inputs[1])[i];
    upper_bounds_[i] = reinterpret_cast<double*>(taskData->inputs[2])[i];
    intervals_[i] = reinterpret_cast<int*>(taskData->inputs[3])[i];
  }
  result_ = 0.0;
  return true;
}

bool fyodorov_m_trapezoidal_method_seq::TestTaskSequential::validation() {
  internal_order_test();
  if (taskData->outputs_count.size() != 1 || taskData->outputs_count[0] != 1) {
    return false;
  }

  // Проверка на корректные границы интегрирования
  for (size_t i = 0; i < lower_bounds_.size(); ++i) {
    if (lower_bounds_[i] >= upper_bounds_[i]) {
      return false;
    }
  }

  // Проверка на корректные интервалы
  for (size_t i = 0; i < intervals_.size(); ++i) {
    if (intervals_[i] <= 0) {
      return false;
    }
  }

  return true;
}

bool fyodorov_m_trapezoidal_method_seq::TestTaskSequential::run() {
  internal_order_test();

  size_t dim = lower_bounds_.size();
  std::vector<double> step(dim);
  for (size_t i = 0; i < dim; ++i) step[i] = (upper_bounds_[i] - lower_bounds_[i]) / intervals_[i];

  std::function<double(std::vector<double>, size_t)> integrate = [&](std::vector<double> current_point,
                                                                     size_t current_dim) -> double {
    if (current_dim == dim) {
      return evaluate_func(func_, current_point);
    }
    double sum = 0.0;
    for (int i = 0; i <= intervals_[current_dim]; ++i) {
      current_point[current_dim] = lower_bounds_[current_dim] + i * step[current_dim];
      double value;
      if (i == 0 || i == intervals_[current_dim]) {
        value = integrate(current_point, current_dim + 1);
      } else {
        value = 2 * integrate(current_point, current_dim + 1);
      }
      sum += value;
    }
    return sum * step[current_dim] / 2.0;
  };
  std::vector<double> start_point(dim, 0.0);
  result_ = integrate(start_point, 0);
  return true;
}

bool fyodorov_m_trapezoidal_method_seq::TestTaskSequential::post_processing() {
  std::cout << "pre_processing() called" << std::endl;
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  return true;
}