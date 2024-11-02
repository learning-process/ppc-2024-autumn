#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "seq/malyshev_v_monte_carlo_integration/include/malyshev_v_monte_carlo_integration.hpp"

using namespace malyshev_v_monte_carlo_integration;

TEST(malyshev_v_monte_carlo_integration, BasicTest) {
  double a = 0.0;
  double b = 1.0;
  int num_points = 100000;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&num_points));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&output));

  MonteCarloIntegration task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  double expected_result = 1.0 / 3.0;
  ASSERT_NEAR(output, expected_result, 0.01);
}

TEST(malyshev_v_monte_carlo_integration, BasicTest2) {
  double a = -1.0;
  double b = 1.0;
  int num_points = 100000;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&num_points));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&output));

  MonteCarloIntegration task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  double expected_result = 2.0 / 3.0;
  ASSERT_NEAR(output, expected_result, 0.01);
}

TEST(malyshev_v_monte_carlo_integration, LargeRangeTest) {
  double a = 0.0;
  double b = 100.0;
  int num_points = 1000000;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&num_points));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&output));

  MonteCarloIntegration task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();

  double expected_result = 333333.3333;
  ASSERT_NEAR(output, expected_result, 1000.0);
}

TEST(malyshev_v_monte_carlo_integration, InputSizeLessThan3) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.0;
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&output));

  MonteCarloIntegration task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(malyshev_v_monte_carlo_integration, InputSizeMoreThan3) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.0;
  int num_points = 100000;
  double extra_input = 5.0;
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&num_points));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&extra_input));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&output));

  MonteCarloIntegration task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(malyshev_v_monte_carlo_integration, OutputSizeLessThan1) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.0;
  int num_points = 100000;
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&num_points));

  MonteCarloIntegration task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(malyshev_v_monte_carlo_integration, OutputSizeMoreThan1) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.0;
  int num_points = 100000;
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&num_points));
  double output1 = 0.0;
  double output2 = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&output1));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&output2));

  MonteCarloIntegration task(taskData);
  ASSERT_FALSE(task.validation());
}
