#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <numbers>
#include <vector>

#include "seq/naumov_b_simpson_method/include/ops_seq.hpp"

TEST(naumov_b_simpson_method_seq, linear_function) {
  auto func = [](double x) { return 2 * x + 1; };
  naumov_b_simpson_method_seq::bound_t bounds = {0.0, 2.0};
  int num_steps = 4;
  double expected = 6.0;  // Интеграл от 2x + 1 на [0, 2] = [x^2 + x] = 6

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.first));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.second));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_seq::TestTaskSequential task(taskData, func, bounds, num_steps);

  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(result, expected, 1e-5);
}

TEST(naumov_b_simpson_method_seq, quadratic_function) {
  auto func = [](double x) { return x * x; };
  naumov_b_simpson_method_seq::bound_t bounds = {0.0, 3.0};
  int num_steps = 6;
  double expected = 9.0;  // Интеграл от x^2 на [0, 3] = x^3/3 = 9

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.first));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.second));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_seq::TestTaskSequential task(taskData, func, bounds, num_steps);

  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(result, expected, 1e-5);
}

TEST(naumov_b_simpson_method_seq, cubic_function) {
  auto func = [](double x) { return x * x * x; };
  naumov_b_simpson_method_seq::bound_t bounds = {0.0, 2.0};
  int num_steps = 6;
  double expected = 4.0;  // Интеграл от x^3 на [0, 2] = x^4/4 = 4

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.first));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.second));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_seq::TestTaskSequential task(taskData, func, bounds, num_steps);

  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(result, expected, 1e-5);
}

TEST(naumov_b_simpson_method_seq, sine_function) {
  auto func = [](double x) { return sin(x); };
  naumov_b_simpson_method_seq::bound_t bounds = {0.0, std::numbers::pi};
  int num_steps = 8;
  double expected = 2.0;  // Интеграл от sin(x) на [0, π] = -cos(x) = 2

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.first));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.second));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_seq::TestTaskSequential task(taskData, func, bounds, num_steps);

  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(result, expected, 1e-5);
}

TEST(naumov_b_simpson_method_seq, invalid_bounds) {
  auto func = [](double x) { return x * x; };
  naumov_b_simpson_method_seq::bound_t bounds = {2.0, 2.0};
  int num_steps = 4;

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.first));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.second));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_seq::TestTaskSequential task(taskData, func, bounds, num_steps);

  ASSERT_FALSE(task.validation());
}

TEST(naumov_b_simpson_method_seq, invalid_steps) {
  auto func = [](double x) { return x * x; };
  naumov_b_simpson_method_seq::bound_t bounds = {0.0, 2.0};
  int num_steps = 3;

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.first));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.second));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_seq::TestTaskSequential task(taskData, func, bounds, num_steps);

  ASSERT_FALSE(task.validation());
}

TEST(naumov_b_simpson_method_seq, invalid_function) {
  naumov_b_simpson_method_seq::bound_t bounds = {0.0, 2.0};
  int num_steps = 3;
  naumov_b_simpson_method_seq::func_1d_t func = nullptr;

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.first));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds.second));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_seq::TestTaskSequential task(taskData, func, bounds, num_steps);

  ASSERT_FALSE(task.validation());
}
