#include <gtest/gtest.h>

#include <limits>
#include <random>
#include <vector>

#include "seq/anufriev_d_max_of_vector_elements/include/ops_seq_anufriev.hpp"

TEST(anufriev_d_max_of_vector_elements, regularVector) {
  std::vector<int32_t> input = {1, 2, 3, -5, 3, 43};
  int32_t expected = 43;
  int32_t actual = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  anufriev_d_max_of_vector_elements_seq::VectorMaxSeq vectorMaxSeq(taskData);
  ASSERT_TRUE(vectorMaxSeq.validation());
  vectorMaxSeq.pre_processing();
  vectorMaxSeq.run();
  vectorMaxSeq.post_processing();
  ASSERT_EQ(expected, actual);
}

TEST(anufriev_d_max_of_vector_elements, positiveNumbers) {
  std::vector<int32_t> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int32_t expected = 10;
  int32_t actual = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  anufriev_d_max_of_vector_elements_seq::VectorMaxSeq vectorMaxSeq(taskData);
  ASSERT_TRUE(vectorMaxSeq.validation());
  vectorMaxSeq.pre_processing();
  vectorMaxSeq.run();
  vectorMaxSeq.post_processing();
  ASSERT_EQ(expected, actual);
}

TEST(anufriev_d_max_of_vector_elements, negativeNumbers) {
  std::vector<int32_t> input = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
  int32_t expected = -1;
  int32_t actual = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  anufriev_d_max_of_vector_elements_seq::VectorMaxSeq vectorMaxSeq(taskData);
  ASSERT_TRUE(vectorMaxSeq.validation());
  vectorMaxSeq.pre_processing();
  vectorMaxSeq.run();
  vectorMaxSeq.post_processing();
  ASSERT_EQ(expected, actual);
}

TEST(anufriev_d_max_of_vector_elements, zeroVector) {
  std::vector<int32_t> input = {0, 0, 0, 0};
  int32_t expected = 0;
  int32_t actual = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  anufriev_d_max_of_vector_elements_seq::VectorMaxSeq vectorMaxSeq(taskData);
  ASSERT_TRUE(vectorMaxSeq.validation());
  vectorMaxSeq.pre_processing();
  vectorMaxSeq.run();
  vectorMaxSeq.post_processing();
  ASSERT_EQ(expected, actual);
}

TEST(anufriev_d_max_of_vector_elements, randomVector) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-1000, 1000);

  std::vector<int32_t> input_vector(50000);
  std::generate(input_vector.begin(), input_vector.end(), [&]() { return distrib(gen); });

  int32_t expected_max = *std::max_element(input_vector.begin(), input_vector.end());

  int32_t actual_max = std::numeric_limits<int32_t>::min();
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input_vector.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual_max));

  anufriev_d_max_of_vector_elements_seq::VectorMaxSeq vectorMaxSeq(taskData);
  ASSERT_TRUE(vectorMaxSeq.validation());
  vectorMaxSeq.pre_processing();
  vectorMaxSeq.run();
  vectorMaxSeq.post_processing();

  ASSERT_EQ(expected_max, actual_max);
}

TEST(anufriev_d_max_of_vector_elements, emptyVector) {
  std::vector<int32_t> input = {};
  int32_t actual = std::numeric_limits<int32_t>::min();

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->outputs_count.emplace_back(1);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&actual));

  anufriev_d_max_of_vector_elements_seq::VectorMaxSeq vectorMaxSeq(taskData);
  ASSERT_TRUE(vectorMaxSeq.validation());
  vectorMaxSeq.pre_processing();
  vectorMaxSeq.run();
  vectorMaxSeq.post_processing();
  ASSERT_EQ(std::numeric_limits<int32_t>::min(), actual);
}

TEST(anufriev_d_max_of_vector_elements, validationNotPassed) {
  std::vector<int32_t> input = {1, 2, 3, -5};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(input.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));

  anufriev_d_max_of_vector_elements_seq::VectorMaxSeq vectorMaxSeq(taskData);
  ASSERT_FALSE(vectorMaxSeq.validation());
}