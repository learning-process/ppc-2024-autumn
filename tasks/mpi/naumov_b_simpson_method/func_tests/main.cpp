// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#include "mpi/naumov_b_simpson_method/include/ops_mpi.hpp"

TEST(naumov_b_simpson_method_mpi, exponential_function_) {
  auto func = [](double x) { return exp(x); };
  double lower_bound = 0.0;
  double upper_bound = 1.0;
  int num_steps = 3000;
  double expected = exp(1) - 1;

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_mpi::TestMPITaskParallel task(taskData, func, lower_bound, upper_bound, num_steps);

  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(result, expected, 1e-5);
}

TEST(naumov_b_simpson_method_mpi, forx_function_) {
  auto func = [](double x) { return x * x * x * x; };
  double lower_bound = 0.0;
  double upper_bound = 1.0;
  int num_steps = 3000;
  double expected = 1.0 / 5.0;

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_mpi::TestMPITaskParallel task(taskData, func, lower_bound, upper_bound, num_steps);

  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(result, expected, 1e-5);
}

TEST(naumov_b_simpson_method_mpi, cubic_function_) {
  auto func = [](double x) { return x * x * x; };
  double lower_bound = 0.0;
  double upper_bound = 2.0;
  int num_steps = 3000;
  double expected = 4.0;

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_mpi::TestMPITaskParallel task(taskData, func, lower_bound, upper_bound, num_steps);

  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(result, expected, 1e-5);
}

TEST(naumov_b_simpson_method_mpi, linear_function_) {
  auto func = [](double x) { return 2 * x + 1; };
  double lower_bound = 0.0;
  double upper_bound = 2.0;
  int num_steps = 3000;
  double expected = 6.0;

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_mpi::TestMPITaskParallel task(taskData, func, lower_bound, upper_bound, num_steps);

  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(result, expected, 1e-5);
}

TEST(naumov_b_simpson_method_mpi, sine_function_) {
  auto func = [](double x) { return sin(x); };
  double lower_bound = 0.0;
  double upper_bound = 3.141592653589793;
  int num_steps = 3000;
  double expected = 2.0;

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_mpi::TestMPITaskParallel task(taskData, func, lower_bound, upper_bound, num_steps);

  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(result, expected, 1e-5);
}

TEST(naumov_b_simpson_method_mpi, quadratic_function_) {
  auto func = [](double x) { return x * x; };
  double lower_bound = 0.0;
  double upper_bound = 3.0;
  int num_steps = 3000;
  double expected = 9.0;

  double result = 0.0;
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(2);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  taskData->inputs_count.emplace_back(1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_steps));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));

  naumov_b_simpson_method_mpi::TestMPITaskParallel task(taskData, func, lower_bound, upper_bound, num_steps);

  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  ASSERT_NEAR(result, expected, 1e-5);
}