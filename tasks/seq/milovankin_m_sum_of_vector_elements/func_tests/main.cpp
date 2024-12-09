#include <gtest/gtest.h>

#include <vector>

#include "seq/milovankin_m_sum_of_vector_elements/include/ops_seq.hpp"

TEST(milovankin_m_sum_of_vector_elements, regularVector) {
  std::vector<int32_t> input = {1, 2, 3, -5, 3, 43};
  int64_t expected = 47;
  int64_t actual = 0;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  milovankin_m_sum_of_vector_elements_seq::VectorSumSeq vectorSumSeq(taskData);
  ASSERT_TRUE(vectorSumSeq.validation());
  vectorSumSeq.pre_processing();
  vectorSumSeq.run();
  vectorSumSeq.post_processing();
  ASSERT_EQ(expected, actual);
}

TEST(milovankin_m_sum_of_vector_elements, positiveNumbers) {
  std::vector<int32_t> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int64_t expected = 55;
  int64_t actual = 0;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  milovankin_m_sum_of_vector_elements_seq::VectorSumSeq vectorSumSeq(taskData);
  ASSERT_TRUE(vectorSumSeq.validation());
  vectorSumSeq.pre_processing();
  vectorSumSeq.run();
  vectorSumSeq.post_processing();
  ASSERT_EQ(expected, actual);
}

TEST(milovankin_m_sum_of_vector_elements, negativeNumbers) {
  std::vector<int32_t> input = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
  int64_t expected = -55;
  int64_t actual = 0;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  milovankin_m_sum_of_vector_elements_seq::VectorSumSeq vectorSumSeq(taskData);
  ASSERT_TRUE(vectorSumSeq.validation());
  vectorSumSeq.pre_processing();
  vectorSumSeq.run();
  vectorSumSeq.post_processing();
  ASSERT_EQ(expected, actual);
}

TEST(milovankin_m_sum_of_vector_elements, zeroVector) {
  std::vector<int32_t> input = {0, 0, 0, 0};
  int64_t expected = 0;
  int64_t actual = 0;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  milovankin_m_sum_of_vector_elements_seq::VectorSumSeq vectorSumSeq(taskData);
  ASSERT_TRUE(vectorSumSeq.validation());
  vectorSumSeq.pre_processing();
  vectorSumSeq.run();
  vectorSumSeq.post_processing();
  ASSERT_EQ(expected, actual);
}

TEST(milovankin_m_sum_of_vector_elements, emptyVector) {
  std::vector<int32_t> input = {};
  int64_t expected = 0;
  int64_t actual = 0;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  milovankin_m_sum_of_vector_elements_seq::VectorSumSeq vectorSumSeq(taskData);
  ASSERT_TRUE(vectorSumSeq.validation());
  vectorSumSeq.pre_processing();
  vectorSumSeq.run();
  vectorSumSeq.post_processing();
  ASSERT_EQ(expected, actual);
}

TEST(milovankin_m_sum_of_vector_elements, validationNotPassed) {
  std::vector<int32_t> input = {1, 2, 3, -5};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));

  milovankin_m_sum_of_vector_elements_seq::VectorSumSeq vectorSumSeq(taskData);
  ASSERT_FALSE(vectorSumSeq.validation());
}
