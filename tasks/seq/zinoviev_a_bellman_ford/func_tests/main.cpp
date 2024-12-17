// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/zinoviev_a_bellman_ford/include/ops_seq.hpp"

TEST(zinoviev_a_bellman_ford, Test_Shortest_Path) {
  const int count = 100;

  // Create data
  std::vector<int> in(1, count);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  zinoviev_a_bellman_ford_seq::BellmanFordTaskSequential bellmanFordTaskSequential(taskDataSeq);
  ASSERT_EQ(bellmanFordTaskSequential.validation(), true);
  bellmanFordTaskSequential.pre_processing();
  bellmanFordTaskSequential.run();
  bellmanFordTaskSequential.post_processing();
  ASSERT_EQ(count, out[0]);
}
