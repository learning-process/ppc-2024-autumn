// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "seq/tsatsyn_a_vector_dot_product/include/ops_seq.hpp"

std::vector<int> toGetRandomVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> v(size);
  for (int i = 0; i < size; i++) {
    v[i] = gen() % 200 + gen() % 10;
  }
  return v;
}

TEST(tsatsyn_a_vector_dot_product_seq, Test_Random_Scalar) {
  const int size = 5;

  // Create data
  std::vector<int> v1 = toGetRandomVector(size);
  std::vector<int> v2 = toGetRandomVector(size);
  std::vector<int> ans(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
  taskDataSeq->inputs_count.emplace_back(v2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
  taskDataSeq->outputs_count.emplace_back(ans.size());

  // Create Task
  tsatsyn_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
}

TEST(tsatsyn_a_vector_dot_product_seq, Test_Negative_Validation) {
  // Create data
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = {1, 2, 3, 4};
  std::vector<int> ans(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
  taskDataSeq->inputs_count.emplace_back(v2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
  taskDataSeq->outputs_count.emplace_back(ans.size());

  // Create Task
  tsatsyn_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(tsatsyn_a_vector_dot_product_seq, Test_Scalar_32) {
  // Create data
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = {4, 5, 6};
  std::vector<int> ans(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
  taskDataSeq->inputs_count.emplace_back(v2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
  taskDataSeq->outputs_count.emplace_back(ans.size());

  // Create Task
  tsatsyn_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(tsatsyn_a_vector_dot_product_seq::resulting(v1, v2), ans[0]);
}

TEST(tsatsyn_a_vector_dot_product_seq, Test_Scalar_28) {
  // Create data
  std::vector<int> v1 = {6, 4, 5};
  std::vector<int> v2 = {1, 3, 2};
  std::vector<int> ans(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
  taskDataSeq->inputs_count.emplace_back(v2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
  taskDataSeq->outputs_count.emplace_back(ans.size());

  // Create Task
  tsatsyn_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(tsatsyn_a_vector_dot_product_seq::resulting(v1, v2), ans[0]);
}

TEST(tsatsyn_a_vector_dot_product_seq, Test_Scalar_95) {
  // Create data
  std::vector<int> v1 = {7, 4, 6};
  std::vector<int> v2 = {3, 5, 9};
  std::vector<int> ans(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
  taskDataSeq->inputs_count.emplace_back(v2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
  taskDataSeq->outputs_count.emplace_back(ans.size());

  // Create Task
  tsatsyn_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(tsatsyn_a_vector_dot_product_seq::resulting(v1, v2), ans[0]);
}

TEST(tsatsyn_a_vector_dot_product_seq, Test_Scalar_2330) {
  // Create data
  std::vector<int> v1 = {20, 54, 23};
  std::vector<int> v2 = {32, 10, 50};
  std::vector<int> ans(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
  taskDataSeq->inputs_count.emplace_back(v2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
  taskDataSeq->outputs_count.emplace_back(ans.size());

  // Create Task
  tsatsyn_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(tsatsyn_a_vector_dot_product_seq::resulting(v1, v2), ans[0]);
}

TEST(tsatsyn_a_vector_dot_product_seq, Test_Scalar_1956) {
  // Create data
  std::vector<int> v1 = {12, 100, 50};
  std::vector<int> v2 = {13, 3, 30};
  std::vector<int> ans(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
  taskDataSeq->inputs_count.emplace_back(v2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
  taskDataSeq->outputs_count.emplace_back(ans.size());

  // Create Task
  tsatsyn_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(tsatsyn_a_vector_dot_product_seq::resulting(v1, v2), ans[0]);
}
