#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "seq/vavilov_v_bellman_ford/include/ops_seq.hpp"

TEST(vavilov_v_bellman_ford_seq, ValidInputWithMultiplePaths_1) {
  std::vector<int> edges = {0, 1, 10, 0, 2, 5, 1, 2, 2, 1, 3, 1, 2, 1, 3, 2, 3, 9, 2, 4, 2, 3, 4, 4};
  std::vector<int> output(5);
  unsigned int vertices = 5, edges_count = 8, source = 0;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count = {vertices, edges_count, 1};
  taskDataSeq->inputs = {reinterpret_cast<uint8_t*>(&source), reinterpret_cast<uint8_t*>(edges.data())};
  taskDataSeq->outputs_count = {output.size()};
  taskDataSeq->outputs = {reinterpret_cast<uint8_t*>(output.data())};

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
  unsigned int vertices = 5, edges_count = 6, source = 0;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count = {vertices, edges_count, 1};
  taskDataSeq->inputs = {reinterpret_cast<uint8_t*>(&source), reinterpret_cast<uint8_t*>(edges.data())};
  taskDataSeq->outputs_count = {output.size()};
  taskDataSeq->outputs = {reinterpret_cast<uint8_t*>(output.data())};

  vavilov_v_bellman_ford_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected_output = {0, -1, 2, -2, 1};
  EXPECT_EQ(output, expected_output);
}

TEST(vavilov_v_bellman_ford_seq, NegativeCycle) {
  std::vector<int> edges = {0, 1, 1, 1, 2, -1, 2, 0, -1};
  std::vector<int> output(3);
  unsigned int vertices = 3, edges_count = 3, source = 0;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count = {vertices, edges_count, 1};
  taskDataSeq->inputs = {reinterpret_cast<uint8_t*>(&source), reinterpret_cast<uint8_t*>(edges.data())};
  taskDataSeq->outputs_count = {output.size()};
  taskDataSeq->outputs = {reinterpret_cast<uint8_t*>(output.data())};

  vavilov_v_bellman_ford_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_FALSE(testTaskSequential.run());
}

TEST(vavilov_v_bellman_ford_seq, DisconnectedGraph) {
  std::vector<int> edges = {0, 1, 4, 0, 2, 1, 1, 3, 2};
  std::vector<int> output(5, std::numeric_limits<int>::max());
  unsigned int vertices = 5, edges_count = 3, source = 0;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count = {vertices, edges_count, 1};
  taskDataSeq->inputs = {reinterpret_cast<uint8_t*>(&source), reinterpret_cast<uint8_t*>(edges.data())};
  taskDataSeq->outputs_count = {output.size()};
  taskDataSeq->outputs = {reinterpret_cast<uint8_t*>(output.data())};

  vavilov_v_bellman_ford_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected_output = {0, 4, 1, 6, std::numeric_limits<int>::max()};
  EXPECT_EQ(output, expected_output);
}

TEST(vavilov_v_bellman_ford_seq, SingleVertexGraph) {
  std::vector<int> edges = {};
  std::vector<int> output(1, 0);
  unsigned int vertices = 1, edges_count = 0, source = 0;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count = {vertices, edges_count, 1};
  taskDataSeq->inputs = {reinterpret_cast<uint8_t*>(&source), reinterpret_cast<uint8_t*>(edges.data())};
  taskDataSeq->outputs_count = {output.size()};
  taskDataSeq->outputs = {reinterpret_cast<uint8_t*>(output.data())};

  vavilov_v_bellman_ford_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected_output = {0};
  EXPECT_EQ(output, expected_output);
}
