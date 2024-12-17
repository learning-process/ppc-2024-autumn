// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/zinoviev_a_bellman_ford/include/ops_seq.hpp"

TEST(zinoviev_a_bellman_ford, Test_Small_Graph) {
  const int num_vertices = 4;
  const int num_edges = 5;
  std::vector<int> graph = {0, 1, 1, 0, 2, 4, 1, 2, 2, 1, 3, 5, 2, 3, 1};
  std::vector<int> dist(num_vertices, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(graph.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(dist.data()));
  taskDataSeq->outputs_count.emplace_back(dist.size());

  zinoviev_a_bellman_ford::BellmanFordSeqTaskSequential testSeqTaskSequential(taskDataSeq);
  ASSERT_EQ(testSeqTaskSequential.validation(), true);
  testSeqTaskSequential.pre_processing();
  testSeqTaskSequential.run();
  testSeqTaskSequential.post_processing();

  std::vector<int> reference_dist = {0, 1, 3, 4};
  ASSERT_EQ(reference_dist, dist);
}

TEST(zinoviev_a_bellman_ford, Test_Medium_Graph) {
  const int num_vertices = 5;
  const int num_edges = 7;
  std::vector<int> graph = {0, 1, 2, 0, 2, 4, 1, 2, 1, 1, 3, 7, 2, 3, 3, 2, 4, 5, 3, 4, 2};
  std::vector<int> dist(num_vertices, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(graph.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(dist.data()));
  taskDataSeq->outputs_count.emplace_back(dist.size());

  zinoviev_a_bellman_ford::BellmanFordSeqTaskSequential testSeqTaskSequential(taskDataSeq);
  ASSERT_EQ(testSeqTaskSequential.validation(), true);
  testSeqTaskSequential.pre_processing();
  testSeqTaskSequential.run();
  testSeqTaskSequential.post_processing();

  std::vector<int> reference_dist = {0, 2, 3, 6, 8};
  ASSERT_EQ(reference_dist, dist);
}