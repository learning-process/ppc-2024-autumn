// Copyright 2024 Nesterov Alexander
#include "seq/naumov_b_simpson_method/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace naumov_b_simpson_method_seq {

double integrir_1d(const func_1d_t &func, const bound_t &bound, int num_steps) {
  auto [lower_bound, upper_bound] = bound;
  double step_size = (upper_bound - lower_bound) / num_steps;
  double result = func(lower_bound) + func(upper_bound);

  for (int step_index = 1; step_index < num_steps; ++step_index) {
    double func_arg = lower_bound + (step_index * step_size);
    int weight = (step_index % 2 == 0) ? 2 : 4;
    result += weight * func(func_arg);
  }

  return result * step_size / 3.0;
}

bool TestTaskSequential::validation() {
  internal_order_test();

  // Проверяем входные данные
  if (bounds_.first >= bounds_.second) {
    return false;  // Границы должны быть корректными
  }

  if (num_steps_ < 2 || num_steps_ % 2 != 0) {
    return false;  // Количество шагов должно быть чётным и >= 2
  }

  if (!function_) {
    return false;  // Функция должна быть задана
  }

  return true;
}

bool TestTaskSequential::pre_processing() {
  internal_order_test();

  // Разделение интервала на сегменты
  double step_size = (bounds_.second - bounds_.first) / num_steps_;
  segments_.clear();

  for (int i = 0; i < num_steps_; ++i) {
    double left = bounds_.first + i * step_size;
    double right = left + step_size;
    segments_.emplace_back(left, right);
  }

  return true;
}

bool TestTaskSequential::run() {
  internal_order_test();

  // Выполнение интегрирования по всем сегментам
  result_ = 0.0;
  for (const auto &segment : segments_) {
    result_ += integrir_1d(function_, segment, num_steps_);
  }

  return true;
}

bool TestTaskSequential::post_processing() {
  internal_order_test();

  // Сохранение результата
  *reinterpret_cast<double *>(taskData->outputs[0]) = result_;
  return true;
}

}  // namespace naumov_b_simpson_method_seq
