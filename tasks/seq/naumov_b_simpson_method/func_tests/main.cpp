// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "seq/naumov_b_simpson_method/include/ops_seq.hpp"

TEST(naumov_b_simpson_method_seq, linear_function) {
  auto func = [](double x) { return 2 * x + 1; };
  naumov_b_simpson_method_seq::bound_t bounds = {0.0, 2.0};
  int num_steps = 4;
  double tolerance = 1e-5;
  double expected = 6.0;  // Интеграл от 2x + 1 на [0, 2] = [x^2 + x] = 6

  double result = naumov_b_simpson_method_seq::integrir_1d(func, bounds, num_steps);
  EXPECT_NEAR(result, expected, tolerance);
}

TEST(naumov_b_simpson_method_seq, quadratic_function) {
  auto func = [](double x) { return x * x; };
  naumov_b_simpson_method_seq::bound_t bounds = {0.0, 3.0};
  int num_steps = 6;
  double tolerance = 1e-5;
  double expected = 9.0;  // Интеграл от x^2 на [0, 3] = x^3/3 = 9

  double result = naumov_b_simpson_method_seq::integrir_1d(func, bounds, num_steps);
  EXPECT_NEAR(result, expected, tolerance);
}

TEST(naumov_b_simpson_method_seq, invalid_bounds) {
  auto func = [](double x) { return x * x; };
  naumov_b_simpson_method_seq::bound_t bounds = {2.0, 2.0};
  int num_steps = 4;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  naumov_b_simpson_method_seq::TestTaskSequential task(task_data, func, bounds, num_steps);
  ASSERT_FALSE(task.validation());
}

TEST(naumov_b_simpson_method_seq, invalid_steps) {
  auto func = [](double x) { return x * x; };
  naumov_b_simpson_method_seq::bound_t bounds = {0.0, 2.0};
  int num_steps = 3;  // Нечётное число шагов

  auto task_data = std::make_shared<ppc::core::TaskData>();
  naumov_b_simpson_method_seq::TestTaskSequential task(task_data, func, bounds, num_steps);
  ASSERT_FALSE(task.validation());
}

TEST(naumov_b_simpson_method_seq, invalid_function) {
  naumov_b_simpson_method_seq::bound_t bounds = {0.0, 2.0};
  int num_steps = 4;
  naumov_b_simpson_method_seq::func_1d_t func = nullptr;  // Некорректная функция

  auto task_data = std::make_shared<ppc::core::TaskData>();
  naumov_b_simpson_method_seq::TestTaskSequential task(task_data, func, bounds, num_steps);
  ASSERT_FALSE(task.validation());
}
