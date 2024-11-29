#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "core/task/include/task.hpp"
#include "seq/malyshev_v_monte_carlo_integration/include/ops_seq.hpp"

TEST(malyshev_v_monte_carlo_integration_sequential, LinearFunctionTest) {
  std::vector<double> global_result(1, 0.0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;
  auto linear_function = [](double x) { return x; };

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq, linear_function);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  double expected_value = 0.5;
  ASSERT_NEAR(global_result[0], expected_value, epsilon);
}

TEST(malyshev_v_monte_carlo_integration_sequential, CubicFunctionTest) {
  std::vector<double> global_result(1, 0.0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;
  auto cubic_function = [](double x) { return x * x * x; };

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq, cubic_function);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  double expected_value = 0.25;
  ASSERT_NEAR(global_result[0], expected_value, epsilon);
}

TEST(malyshev_v_monte_carlo_integration_sequential, SineFunctionTest) {
  std::vector<double> global_result(1, 0.0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = M_PI;
  double epsilon = 0.0004;
  auto sine_function = [](double x) { return std::sin(x); };

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq, sine_function);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  double expected_value = 2.0;
  ASSERT_NEAR(global_result[0], expected_value, epsilon);
}

TEST(malyshev_v_monte_carlo_integration_sequential, ExponentialFunctionTest) {
  std::vector<double> global_result(1, 0.0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;
  auto exponential_function = [](double x) { return std::exp(x); };

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq, exponential_function);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  double expected_value = std::exp(1.0) - 1;
  ASSERT_NEAR(global_result[0], expected_value, epsilon);
}
