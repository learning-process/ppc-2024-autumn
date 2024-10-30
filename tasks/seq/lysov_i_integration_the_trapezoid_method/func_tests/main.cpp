#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "seq/lysov_i_integration_the_trapezoid_method/include/ops_seq.hpp"

TEST(lysov_i_integration_the_trapezoid_method_seq, BasicTest) {
  double a = 0.0;
  double b = 1.45;
  double epsilon = 1e-2;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  double expected_result = 1.016;
  ASSERT_NEAR(output, expected_result, epsilon);
}

TEST(lysov_i_integration_the_trapezoid_method_seq, BasicTest2) {
  double a = -1.45;
  double b = 0.0;
  double epsilon = 0.01;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  double expected_result = 1.016;
  std::cout << output << std::endl;
  ASSERT_NEAR(output, expected_result, epsilon);
}

TEST(lysov_i_integration_the_trapezoid_method_seq, BasicTest3) {
  double a = -1.45;
  double b = 1.45;
  double epsilon = 0.01;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  double expected_result = 2.03;
  ASSERT_NEAR(output, expected_result, epsilon);
}

TEST(lysov_i_integration_the_trapezoid_method_seq, BasicTest4) {
  double a = 1.45;
  double b = 0;
  double epsilon = 0.01;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  double expected_result = -1.016;
  ASSERT_NEAR(output, expected_result, epsilon);
}

TEST(lysov_i_integration_the_trapezoid_method_seq, BasicTest5) {
  double a = 0.0;
  double b = 100.0;
  double epsilon = 0.001;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  double expected_result =333333.333510;
  std::cout << output << std::endl;
  ASSERT_NEAR(output, expected_result, epsilon);
}

TEST(lysov_i_integration_the_trapezoid_method_seq, BasicTest6) {
  double a = -10.0;
  double b = 65.0;
  double epsilon = 0.01;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  double expected_result = 91875.001;  //
  std::cout << output << std::endl;
  ASSERT_NEAR(output, expected_result, epsilon);
}

TEST(lysov_i_integration_the_trapezoid_method_seq, BasicTest7) {
  double a = -10.0;
  double b = 10.0;
  double epsilon = 0.001;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
  double output = 0.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  std::cout << output << std::endl;
  double expected_result = 666.66666;  //
  std::cout << output << std::endl;
  ASSERT_NEAR(output, expected_result, 1e-2);
}

TEST(lysov_i_integration_the_trapezoid_method_seq, InputSizeLessThan3) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  double a = -1.0;
  double b = 1.0;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  double result = 0.0;
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(lysov_i_integration_the_trapezoid_method_seq, InputSizeMoreThan3) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  double a = -1.0;
  double b = 1.0;
  double epsilon = 0.01;
  double extra_input = 5.0;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&extra_input));
  double result = 0.0;
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(lysov_i_integration_the_trapezoid_method_seq, OutputSizeLessThan1) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  double a = -1.0;
  double b = 1.0;
  double epsilon = 0.01;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(lysov_i_integration_the_trapezoid_method_seq, OutputSizeMoreThan1) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  double a = -1.0;
  double b = 1.0;
  double epsilon = 0.01;
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  double result1 = 0.0;
  double result2 = 0.0;
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result1));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result2));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}
