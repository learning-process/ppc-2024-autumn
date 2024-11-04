// Golovkin Maksim

#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"
#include "seq/golovkin_integration_rectangular_method/include/ops_seq.hpp"
using namespace golovkin_integration_rectangular_method;
using ppc::core::TaskData;

TEST(golovkin_integration_rectangular_method_seq, Calculate_ZeroToOne) {
  auto taskData = std::make_shared<TaskData>();
  double a = 0.0, b = 1.0, epsilon = 0.001;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));
  taskData->state_of_testing = TaskData::StateOfTesting::FUNC;

  IntegralCalculator calculator(taskData);

  ASSERT_TRUE(calculator.validation());
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, 1.0 / 3.0, 0.01);
}

TEST(golovkin_integration_rectangular_method_seq, Calculate_ZeroToTwo) {
  auto taskData = std::make_shared<TaskData>();
  double a = 0.0, b = 2.0, epsilon = 0.001;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));
  taskData->state_of_testing = TaskData::StateOfTesting::FUNC;

  IntegralCalculator calculator(taskData);

  ASSERT_TRUE(calculator.validation());
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, 8.0 / 3.0, 0.01);
}

TEST(golovkin_integration_rectangular_method_seq, Calculate_NegativeToPositive) {
  auto taskData = std::make_shared<TaskData>();
  double a = -1.0, b = 1.0, epsilon = 0.001;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));
  taskData->state_of_testing = TaskData::StateOfTesting::FUNC;

  IntegralCalculator calculator(taskData);

  ASSERT_TRUE(calculator.validation());
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, 2.0 / 3.0, 0.01);
}

TEST(golovkin_integration_rectangular_method_seq, Calculate_LargeInterval_HighPrecision) {
  auto taskData = std::make_shared<TaskData>();
  double a = 0.0, b = 10.0, epsilon = 0.00001;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));

  IntegralCalculator calculator(taskData);

  ASSERT_TRUE(calculator.validation());
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, 333.3333333333333, 0.01);
}

TEST(golovkin_integration_rectangular_method_seq, Calculate_OneToThree_HighPrecision) {
  auto taskData = std::make_shared<TaskData>();
  double a = 1.0, b = 3.0, epsilon = 0.0001;
  double result = 0.0;

  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskData->inputs_count = {sizeof(double), sizeof(double), sizeof(double)};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskData->outputs_count.push_back(sizeof(double));
  taskData->state_of_testing = TaskData::StateOfTesting::FUNC;

  IntegralCalculator calculator(taskData);

  ASSERT_TRUE(calculator.validation());
  ASSERT_TRUE(calculator.pre_processing());
  ASSERT_TRUE(calculator.run());
  ASSERT_TRUE(calculator.post_processing());

  EXPECT_NEAR(result, 26.0 / 3.0, 0.01);
}