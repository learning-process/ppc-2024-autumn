#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "seq/vavilov_v_bellman_ford/include/ops_seq.hpp"

TEST(vavilov_v_bellman_ford_seq, ValidInputWithMultiplePaths_1) {
  std::vector<int> edges = {0, 1, 10, 0, 2, 5, 1, 2, 2, 1, 3, 1, 2, 1, 3, 2, 3, 9, 2, 4, 2, 3, 4, 4};
  std::vector<int> output(5);
  unsigned int vertices = 5;
  unsigned int edges_count = 8;
  unsigned int source = 0;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(vertices);
  taskDataSeq->inputs_count.emplace_back(edges_count);
  taskDataSeq->inputs_count.emplace_back(source);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  vavilov_v_bellman_ford_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected_output = {0, 8, 5, 9, 7};
  EXPECT_EQ(output, expected_output);
}

TEST(vavilov_v_bellman_ford_seq, ValidInputWithMultiplePaths_2) {
  std::vector<int> edges = {0, 1, -1, 0, 2, 4, 1, 2, 3, 1, 3, 2, 2, 3, 5, 3, 4, -3};
  std::vector<int> output(5);
  unsigned int vertices = 5;
  unsigned int edges_count = 6;
  unsigned int source = 0;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(vertices);
  taskDataSeq->inputs_count.emplace_back(edges_count);
  taskDataSeq->inputs_count.emplace_back(source);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  vavilov_v_bellman_ford_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected_output = {0, -1, 2, 1, -2};
  EXPECT_EQ(output, expected_output);
}

TEST(vavilov_v_bellman_ford_seq, DisconnectedGraph) {
  std::vector<int> edges = {0, 1, 4, 0, 2, 1, 1, 3, 2};
  std::vector<int> output(5);
  unsigned int vertices = 5;
  unsigned int edges_count = 3;
  unsigned int source = 0;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(vertices);
  taskDataSeq->inputs_count.emplace_back(edges_count);
  taskDataSeq->inputs_count.emplace_back(source);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  vavilov_v_bellman_ford_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected_output = {0, 4, 1, 6, INT_MAX};
  EXPECT_EQ(output, expected_output);
}

TEST(vavilov_v_bellman_ford_seq, NegativeCycle) {
  std::vector<int> edges = {0, 1, 1, 1, 2, -1, 2, 0, -1};
  std::vector<int> output(3);
  unsigned int vertices = 3;
  unsigned int edges_count = 3;
  unsigned int source = 0;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(vertices);
  taskDataSeq->inputs_count.emplace_back(edges_count);
  taskDataSeq->inputs_count.emplace_back(source);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  vavilov_v_bellman_ford_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_FALSE(testTaskSequential.run());
}

TEST(vavilov_v_bellman_ford_seq, SingleVertexGraph) {
  std::vector<int> edges = {};
  std::vector<int> output(1, 0);
  unsigned int vertices = 1;
  unsigned int edges_count = 0;
  unsigned int source = 0;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(vertices);
  taskDataSeq->inputs_count.emplace_back(edges_count);
  taskDataSeq->inputs_count.emplace_back(source);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  vavilov_v_bellman_ford_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected_output = {0};
  EXPECT_EQ(output, expected_output);
}
