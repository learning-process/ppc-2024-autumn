// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/titov_s_vector_sum/include/ops_seq.hpp"

TEST(titov_s_vector_sum_seq, Test_Int) {
  // Create data
  std::vector<int32_t> in(1, 10);
  const int expected_sum = 10;
  std::vector<int32_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  titov_s_vector_sum_seq::VectorSumSequential<int32_t> vectorSumSequential(taskDataSeq);
  ASSERT_TRUE(vectorSumSequential.validation());
  vectorSumSequential.pre_processing();
  vectorSumSequential.run();
  vectorSumSequential.post_processing();
  ASSERT_EQ(expected_sum, out[0]);
}

TEST(titov_s_vector_sum_seq, Test_Double) {
  // Create data
  std::vector<double> in(1, 10);
  const int expected_sum = 10;
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  titov_s_vector_sum_seq::VectorSumSequential<double> vectorSumSequential(taskDataSeq);
  ASSERT_TRUE(vectorSumSequential.validation());
  vectorSumSequential.pre_processing();
  vectorSumSequential.run();
  vectorSumSequential.post_processing();
  EXPECT_NEAR(out[0], expected_sum, 1e-6);
}

TEST(titov_s_vector_sum_seq, Test_Float) {
  // Create data
  std::vector<float> in(1, 1.f);
  std::vector<float> out(1, 0.f);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  titov_s_vector_sum_seq::VectorSumSequential<float> vectorSumSequential(taskDataSeq);
  ASSERT_TRUE(vectorSumSequential.validation());
  vectorSumSequential.pre_processing();
  vectorSumSequential.run();
  vectorSumSequential.post_processing();
  EXPECT_NEAR(out[0], static_cast<float>(in.size()), 1e-3f);
}

TEST(titov_s_vector_sum_seq, Test_Int64_t) {
  // Create data
  std::vector<int64_t> in(75836, 1);
  std::vector<int64_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  titov_s_vector_sum_seq::VectorSumSequential<int64_t> vectorSumSequential(taskDataSeq);
  ASSERT_TRUE(vectorSumSequential.validation());
  vectorSumSequential.pre_processing();
  vectorSumSequential.run();
  vectorSumSequential.post_processing();
  ASSERT_EQ(static_cast<uint64_t>(out[0]), in.size());
}

TEST(titov_s_vector_sum_seq, Test_Uint8_t) {
  // Create data
  std::vector<uint8_t> in(255, 1);
  std::vector<uint8_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  titov_s_vector_sum_seq::VectorSumSequential<uint8_t> vectorSumSequential(taskDataSeq);
  ASSERT_TRUE(vectorSumSequential.validation());
  vectorSumSequential.pre_processing();
  vectorSumSequential.run();
  vectorSumSequential.post_processing();
  ASSERT_EQ(static_cast<uint64_t>(out[0]), in.size());
}
