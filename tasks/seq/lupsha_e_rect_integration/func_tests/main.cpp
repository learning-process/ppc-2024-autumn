// Copyright 2024 Lupsha Egor
#include <gtest/gtest.h>

#include <vector>

#include "seq/lupsha_e_rect_integration/include/ops_seq.hpp"

TEST(lupsha_e_rect_integration_seq, Test_Rect1) {
  double lower_bound = 0.0;
  double upper_bound = 1.0;
  int num_intervals = 4;

  double expected_result = 0.21875;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_intervals));
  taskDataSeq->inputs_count.emplace_back(3);

  std::vector<double> result(1, 0.0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  lupsha_e_rect_integration_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  std::function<double(double)> f = [](double x) { return x * x; };
  TestTaskSequential.function_set(f);
  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_DOUBLE_EQ(expected_result, result[0]);
}

TEST(lupsha_e_rect_integration_seq, Test_Rect2) {
  double lower_bound = 0.0;
  double upper_bound = 4.0;
  int num_intervals = 10000;

  double expected_result = 21.33;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_intervals));
  taskDataSeq->inputs_count.emplace_back(3);

  std::vector<double> result(1, 0.0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  lupsha_e_rect_integration_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  std::function<double(double)> f = [](double x) { return x * x; };
  TestTaskSequential.function_set(f);

  ASSERT_TRUE(TestTaskSequential.validation());
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  EXPECT_NEAR(result[0], expected_result, 0.01);
}