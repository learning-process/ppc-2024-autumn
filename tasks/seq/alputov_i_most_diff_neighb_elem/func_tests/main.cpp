// Copyright 2024 Alputov Ivan
#include <gtest/gtest.h>

#include <random>
#include <utility>
#include <vector>

#include "seq/alputov_i_most_diff_neighb_elem/include/ops_seq.hpp"

TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_Typical) {
  std::vector<int> inputVector = {10, 20, 40, 80, 128, 78, -12, -15, 44, 90, 51};
  int outputPair[2] = {0, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPair));
  taskDataSeq->outputs_count.emplace_back(2);

  alputov_i_most_diff_neighb_elem_seq::SequentialTask testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(outputPair[0], 78);
  ASSERT_EQ(outputPair[1], -12);
}

TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_NegativeValues) {
  std::vector<int> inputVector = {-3, -6, -9, -11};
  int outputPair[2] = {0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPair));
  taskDataSeq->outputs_count.emplace_back(2);

  // Create Task
  alputov_i_most_diff_neighb_elem_seq::SequentialTask testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(outputPair[0], -3);
  ASSERT_EQ(outputPair[1], -6);
}

TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_EqualElements) {
  std::vector<int> inputVector = {2, 2, 2, 2, 2};
  int outputPair[2] = {0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPair));
  taskDataSeq->outputs_count.emplace_back(2);

  // Create Task
  alputov_i_most_diff_neighb_elem_seq::SequentialTask testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(outputPair[0], 2);
  ASSERT_EQ(outputPair[1], 2);
}

TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_TwoElements) {
  std::vector<int> inputVector = {5, 10};
  int outputPair[2] = {0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPair));
  taskDataSeq->outputs_count.emplace_back(2);

  // Create Task
  alputov_i_most_diff_neighb_elem_seq::SequentialTask testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(outputPair[0], 5);
  ASSERT_EQ(outputPair[1], 10);
}

TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_SingleElement) {
  std::vector<int> inputVector = {100};
  int outputPair[2] = {0, 0};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPair));
  taskDataSeq->outputs_count.emplace_back(2);

  // Create Task
  alputov_i_most_diff_neighb_elem_seq::SequentialTask testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_EmptyVector) {
  std::vector<int> inputVector = {};
  int outputPair[2] = {0, 0};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPair));
  taskDataSeq->outputs_count.emplace_back(2);

  // Create Task
  alputov_i_most_diff_neighb_elem_seq::SequentialTask testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}
TEST(alputov_i_most_diff_neighbor_elem_seq, Test_MaxDiff_RandomLargeVector) {
  const int count = 1000000;
  const int fixedSeed = 12345;
  std::mt19937 gen(fixedSeed);
  std::uniform_int_distribution<> dist(-1000, 1000);
  int outputPair[2] = {0, 0};
  std::vector<int> inputVector(count);
  for (int i = 0; i < count; ++i) {
    inputVector[i] = dist(gen);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSeq->inputs_count.emplace_back(inputVector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputPair));
  taskDataSeq->outputs_count.emplace_back(2);

  alputov_i_most_diff_neighb_elem_seq::SequentialTask testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(1000, outputPair[0]);
  ASSERT_EQ(-1000, outputPair[1]);
}