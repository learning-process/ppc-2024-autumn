#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "seq/bessonov_e_integration_monte_carlo/include/ops_seq.hpp"

TEST(bessonov_e_integration_monte_carlo_seq, PositiveRangeTest) {
  double a = 0.0;
  double b = 1.0;
  int num_points = 1000000;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&num_points));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  bessonov_e_integration_monte_carlo_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  double expected_result = 0.25;
  ASSERT_NEAR(output, expected_result, 1e-1);
}

TEST(bessonov_e_integration_monte_carlo_seq, NegativeRangeTest) {
  double a = -1.0;
  double b = 0.0;
  int num_points = 1000000;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&num_points));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  bessonov_e_integration_monte_carlo_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  double expected_result = -0.25;
  ASSERT_NEAR(output, expected_result, 1e-1);
}

TEST(bessonov_e_integration_monte_carlo_seq, FullRangeTest) {
  double a = -1.0;
  double b = 2.0;
  int num_points = 1000000;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&num_points));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  bessonov_e_integration_monte_carlo_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  double expected_result = 3.75;
  ASSERT_NEAR(output, expected_result, 1e-1);
}

TEST(bessonov_e_integration_monte_carlo_seq, InputSizeLessThan3) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.0;
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  bessonov_e_integration_monte_carlo_seq::TestTaskSequential task(taskData);
  ASSERT_FALSE(task.validation());
}

TEST(bessonov_e_integration_monte_carlo_seq, OutputSizeLessThan1) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.0;
  int num_points = 10000;
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&num_points));
  bessonov_e_integration_monte_carlo_seq::TestTaskSequential task(taskData);
  ASSERT_FALSE(task.validation());
}