// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#include "mpi/naumov_b_simpson_method/include/ops_mpi.hpp"

namespace naumov_b_simpson_method_mpi {

TEST(naumov_b_simpson_method_mpi, exponential_function) {
  auto func = [](double x) { return exp(x); };
  double lower_bound = 0.0;
  double upper_bound = 1.0;
  int num_steps = 3000;
  double expected = exp(1) - 1;  // Интеграл от e^x на [0, 1] = e^1 - e^0 = e - 1

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

TEST(naumov_b_simpson_method_mpi, forx_function) {
  auto func = [](double x) { return x * x * x * x; };
  double lower_bound = 0.0;
  double upper_bound = 1.0;
  int num_steps = 3000;
  double expected = 1.0 / 5.0;  // Интеграл от x^4 на [0, 1] = x^5 / 5 = 1/5

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

TEST(naumov_b_simpson_method_mpi, cubic_function) {
  auto func = [](double x) { return x * x * x; };
  double lower_bound = 0.0;
  double upper_bound = 2.0;
  int num_steps = 3000;
  double expected = 4.0;  // Интеграл от x^3 на [0, 2] = x^4/4 = 4

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

TEST(naumov_b_simpson_method_mpi, linear_function) {
  auto func = [](double x) { return 2 * x + 1; };
  double lower_bound = 0.0;
  double upper_bound = 2.0;
  int num_steps = 3000;
  double expected = 6.0;  // Интеграл от 2x + 1 на [0, 2] = [x^2 + x] = 6

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

TEST(naumov_b_simpson_method_mpi, sine_function) {
  auto func = [](double x) { return sin(x); };
  double lower_bound = 0.0;
  double upper_bound = 3.141592653589793;
  int num_steps = 3000;
  double expected = 2.0;  // Интеграл от sin(x) на [0, π] = -cos(x) = 2

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

TEST(naumov_b_simpson_method_mpi, quadratic_function) {
  auto func = [](double x) { return x * x; };
  double lower_bound = 0.0;
  double upper_bound = 3.0;
  int num_steps = 3000;
  double expected = 9.0;  // Интеграл от x^2 на [0, 3] = x^3/3 = 9

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

}  // namespace naumov_b_simpson_method_mpi
