#include <gtest/gtest.h>

#include <vector>

#include "core/task/include/task.hpp"
#include "seq/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

TEST(malyshev_v_monte_carlo_integration, BasicIntegrationTest) {
  std::vector<double> global_result(1, 0.0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  double expected_value = 0.3333;
  ASSERT_NEAR(global_result[0], expected_value, epsilon);
}

TEST(malyshev_v_monte_carlo_integration, LargeRangeIntegrationTest) {
  std::vector<double> global_result(1, 0.0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 10.0;
  double epsilon = 0.0001;

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  double expected_value = 333.3;
  ASSERT_NEAR(global_result[0], expected_value, 0.05);
}

TEST(malyshev_v_monte_carlo_integration, SmallRangeIntegrationTest) {
  std::vector<double> global_result(1, 0.0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 0.1;
  double epsilon = 0.0001;

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  double expected_value = 0.0003333;
  ASSERT_NEAR(global_result[0], expected_value, epsilon);
}

TEST(malyshev_v_monte_carlo_integration, ZeroRangeIntegrationTest) {
  std::vector<double> global_result(1, 0.0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  double a = 1.0;
  double b = 1.0;
  double epsilon = 0.0001;

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  double expected_value = 0.0;
  ASSERT_NEAR(global_result[0], expected_value, epsilon);
}

TEST(malyshev_v_monte_carlo_integration, NegativeRangeIntegrationTest) {
  std::vector<double> global_result(1, 0.0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  double a = -1.0;
  double b = 0.0;
  double epsilon = 0.0004;

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));

  malyshev_v_monte_carlo_integration::TestMPITaskSequential testTask(taskDataSeq);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  double expected_value = 0.3333;
  ASSERT_NEAR(global_result[0], expected_value, epsilon);
}
