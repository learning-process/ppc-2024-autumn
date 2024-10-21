// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/lysov_i_integration_the_trapezoid_method/include/ops_seq.hpp"

#include <gtest/gtest.h>
#include <memory>

TEST(lysov_i_integration_the_trapezoid_method_seq, BasicTest) {
  double a = 0.0;
  double b = 1.45;
  int cnt_of_splits = 100;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&cnt_of_splits));
  double output = 1.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  double expected_result = 1.0;
  ASSERT_NEAR(output, expected_result, 1e-1);  
}

TEST(lysov_i_integration_the_trapezoid_method_seq, BasicTest2) { 
  double a = -1.45;
  double b = 0.0;
  int cnt_of_splits = 100;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&cnt_of_splits));
  double output = 1.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  double expected_result = 1.0;
  ASSERT_NEAR(output, expected_result, 1e-1);
}

TEST(lysov_i_integration_the_trapezoid_method_seq, BasicTest3) {
  double a = -1.45;
  double b = 1.45;
  int cnt_of_splits = 100;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&cnt_of_splits));
  double output = 1.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  lysov_i_integration_the_trapezoid_method_seq::TestTaskSequential task(taskData);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  double expected_result = 2.0;
  ASSERT_NEAR(output, expected_result, 1e-1);
}
