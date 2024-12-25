// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/gnitienko_k_bellman-ford_algorithm/include/ops_seq.hpp"

namespace gnitienko_k_generate_func {
int INF = std::numeric_limits<int>::max();
std::vector<int> generateGraph(int V, int E) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> graph(V * V, 0);
  std::set<std::pair<int, int>> edges;

  std::uniform_int_distribution<> dist_vertex(0, V - 1);
  std::uniform_int_distribution<> dist_weight(-10, 10);

  int edgeCount = 0;
  while (edgeCount < E) {
    int u = dist_vertex(gen);
    int v = dist_vertex(gen);
    if (u != v && u < v && edges.find({u, v}) == edges.end()) {
      int weight = dist_weight(gen);
      graph[u * V + v] = weight;
      edges.insert({u, v});
      edgeCount++;
    }
  }

  return graph;
}
}  // namespace gnitienko_k_generate_func

TEST(gnitienko_k_bellman_ford_algorithm, test_1st_graph) {
  const int V = 6;
  const int E = 8;

  // Create data
  std::vector<int> graph = {0, 10, 0,  0, 0, 8, 0, 0,  0, 2,  0, 0, 0, 1, 0, 0, 0, 0,
                            0, 0,  -2, 0, 0, 0, 0, -4, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0};
  std::vector<int> res(V, 0);
  std::vector<int> expected_res = {0, 5, 5, 7, 9, 8};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_res, res);
}

TEST(gnitienko_k_bellman_ford_algorithm, test_negative_cycle) {
  const int V = 4;
  const int E = 5;

  std::vector<int> graph = {0, 0, -2, 0, 4, 0, -3, 0, 0, 0, 0, 2, 0, -1, 0, 0};

  std::vector<int> res(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  ASSERT_FALSE(testTaskSequential.run());
}

TEST(gnitienko_k_bellman_ford_algorithm, test_empty_graph) {
  const int V = 0;
  const int E = 0;

  std::vector<int> graph = {};

  std::vector<int> res(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(gnitienko_k_bellman_ford_algorithm, test_graph_2) {
  const int V = 5;
  const int E = 6;

  std::vector<int> graph = {0, 10, 5, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0};

  std::vector<int> res(V, 0);
  std::vector<int> expected_res = {0, 10, 5, 11, 14};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  ASSERT_TRUE(testTaskSequential.run());
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_res, res);
}

TEST(gnitienko_k_bellman_ford_algorithm, test_graph_10_12) {
  const int V = 10;
  const int E = 12;

  std::vector<int> graph = {0, 4, -2, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2,
                            0, 0, 0,  0,  0, 0, 0, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0,
                            0, 0, 0,  0,  0, 0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0,  -2, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 3, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int> res(V, 0);
  std::vector<int> expected_res = {0, -1, -2, 1, 0, -4, -3, -2, -4, -1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  ASSERT_TRUE(testTaskSequential.run());
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_res, res);
}

TEST(gnitienko_k_bellman_ford_algorithm, test_random_graph) {
  const int V = 7;
  const int E = 10;

  std::vector<int> graph = gnitienko_k_generate_func::generateGraph(V, E);

  std::vector<int> res(V, 0);
  std::vector<int> expected_res(V, gnitienko_k_generate_func::INF);
  expected_res[0] = 0;

  bool changed = true;
  for (int k = 0; k < V && changed; k++) {
    changed = false;
    for (int i = 0; i < V; ++i) {
      for (int j = 0; j < V; ++j) {
        if (i != j) {
          if (graph[i * V + j] != 0 && expected_res[i] != gnitienko_k_generate_func::INF &&
              graph[i * V + j] + expected_res[i] < expected_res[j]) {
            expected_res[j] = graph[i * V + j] + expected_res[i];
            changed = true;
          }
        }
      }
    }
  }

  for (int i = 0; i < V; i++)
    if (expected_res[i] == gnitienko_k_generate_func::INF) expected_res[i] = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  ASSERT_TRUE(testTaskSequential.run());
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_res, res);
}
