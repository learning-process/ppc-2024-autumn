// Copyright 2023 Nesterov Alexander
#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <utility>  // Для std::pair
#include <vector>

#include "core/task/include/task.hpp"

namespace naumov_b_simpson_method_seq {

using bound_t = std::pair<double, double>;        // Границы интегрирования
using func_1d_t = std::function<double(double)>;  // Тип функции для интеграции

// Функция интегрирования по методу Симпсона
double integrir_1d(const func_1d_t &func, const bound_t &bound, int num_steps);

class TestTaskSequential : public ppc::core::Task {
 public:
  TestTaskSequential(std::shared_ptr<ppc::core::TaskData> task_data, func_1d_t function, bound_t bounds, int num_steps)
      : Task(std::move(task_data)),
        function_(std::move(function)),
        bounds_(std::move(bounds)),
        num_steps_(num_steps),
        result_(0.0) {}

  // Переопределённые методы из базового класса
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  bound_t bounds_;                 // Границы интегрирования [a, b]
  int num_steps_;                  // Количество шагов
  func_1d_t function_;             // Функция для интеграции
  double result_;                  // Результат вычислений
  std::vector<bound_t> segments_;  // Сегменты для интеграции
};

}  // namespace naumov_b_simpson_method_seq
