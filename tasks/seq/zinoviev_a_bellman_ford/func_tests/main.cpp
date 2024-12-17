// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/zinoviev_a_bellman_ford/include/ops_seq.hpp"

TEST(zinoviev_a_bellman_ford, Test_Small_Graph) {
  // Example graph in CRS format
  std::vector<int> graph = {0, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  std::vector<int> distances(4, INT_MAX);
  distances[0] = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(graph.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  taskDataSeq->outputs_count.emplace_back(distances.size());

  // Create Task
  zinoviev_a_bellman_ford_seq::BellmanFordSeqTaskSequential bellmanFordSeqTaskSequential(taskDataSeq);
  ASSERT_EQ(bellmanFordSeqTaskSequential.validation(), true);
  bellmanFordSeqTaskSequential.pre_processing();
  bellmanFordSeqTaskSequential.run();
  bellmanFordSeqTaskSequential.post_processing();

  // Reference distances
  std::vector<int> reference_distances = {0, 2, 3, 1};
  ASSERT_EQ(distances, reference_distances);
}

TEST(zinoviev_a_bellman_ford, Test_Medium_Graph) {
  // Example graph in CRS format
  std::vector<int> graph = {0, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  std::vector<int> distances(6, INT_MAX);
  distances[0] = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(graph.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  taskDataSeq->outputs_count.emplace_back(distances.size());

  // Create Task
  zinoviev_a_bellman_ford_seq::BellmanFordSeqTaskSequential bellmanFordSeqTaskSequential(taskDataSeq);
  ASSERT_EQ(bellmanFordSeqTaskSequential.validation(), true);
  bellmanFordSeqTaskSequential.pre_processing();
  bellmanFordSeqTaskSequential.run();
  bellmanFordSeqTaskSequential.post_processing();

  // Reference distances
  std::vector<int> reference_distances = {0, 2, 3, 1, 2, 3};
  ASSERT_EQ(distances, reference_distances);
}