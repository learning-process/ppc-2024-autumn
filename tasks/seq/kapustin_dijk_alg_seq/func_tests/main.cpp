#include <gtest/gtest.h>

#include "seq/kapustin_dijk_alg_seq/include/seq_alg.hpp"

namespace generateGraph {
std::vector<int> generateGraph(int V, int E) {
  std::uniform_int_distribution<int> vertex_dist(0, V - 1);
  std::uniform_int_distribution<int> weight_dist(1, 10);
  std::random_device rd;
  std::mt19937 rng(rd());
  std::vector<int> graph(V * V, 0);
  int edges_added = 0;
  while (edges_added < E) {
    int src = vertex_dist(rng);
    int dest = vertex_dist(rng);

    if (src != dest && graph[src * V + dest] == 0) {
      int weight = weight_dist(rng);
      graph[src * V + dest] = weight;
      graph[dest * V + src] = weight;
      ++edges_added;
    }
  }

  return graph;
}
}  // namespace generateGraph
TEST(kapustin_dijkstras_algorithm, test_random_graph) {
  const int V = 3;
  const int E = 3;
  const int INF = std::numeric_limits<int>::max();
  std::vector<int> graph = generateGraph::generateGraph(V, E);

  std::vector<int> res(V, 0);

  std::vector<int> expected_res(V, INF);
  expected_res[0] = 0;

  std::set<int> processed;
  while (processed.size() < static_cast<size_t>(V)) {
    int u = -1;

    for (int i = 0; i < V; i++) {
      if (processed.find(i) == processed.end() && (u == -1 || expected_res[i] < expected_res[u])) {
        u = i;
      }
    }

    if (u == -1 || expected_res[u] == INF) {
      break;
    }

    processed.insert(u);
    for (int v = 0; v < V; v++) {
      if (graph[u * V + v] != 0 && processed.find(v) == processed.end()) {
        expected_res[v] = std::min(expected_res[v], expected_res[u] + graph[u * V + v]);
      }
    }
  }

  for (int i = 0; i < V; i++) {
    if (expected_res[i] == INF) {
      expected_res[i] = 0;
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(expected_res, res);
}
TEST(kapustin_dijkstras_algorithm, test_1st_graph) {
  const int V = 3;
  const int E = 3;

  std::vector<int> graph = {0, 1, 4, 1, 0, 2, 4, 2, 0};

  std::vector<int> res(V, 0);

  std::vector<int> expected_res = {0, 1, 3};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_res, res);
}

TEST(kapustin_dijkstras_algorithm, test_2nd_graph) {
  const int V = 5;
  const int E = 8;

  std::vector<int> graph = {0, 10, 0, 0, 0, 10, 0, 5, 0, 0, 0, 5, 0, 20, 0, 0, 0, 20, 0, 1, 0, 0, 0, 1, 0};

  std::vector<int> res(V, 0);

  std::vector<int> expected_res = {0, 10, 15, 35, 36};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());

  testTaskSequential.pre_processing();

  testTaskSequential.run();

  testTaskSequential.post_processing();

  ASSERT_EQ(expected_res, res);
}
TEST(kapustin_dijkstras_algorithm, test_3nd_graph) {
  const int V = 6;
  const int E = 8;

  std::vector<int> graph = {0, 7,  9,  0, 0, 14, 7, 0, 10, 15, 0, 0, 9,  10, 0, 11, 0, 2,
                            0, 15, 11, 0, 6, 0,  0, 0, 0,  6,  0, 9, 14, 0,  2, 0,  9, 0};

  std::vector<int> res(V, 0);

  std::vector<int> expected_res = {0, 7, 9, 20, 20, 11};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(expected_res, res);
}
TEST(kapustin_dijkstras_algorithm, test_4rd_graph) {
  const int V = 7;
  const int E = 9;

  std::vector<int> graph = {0, 2, 0, 5, 0, 0, 0, 2, 0, 3, 4, 0, 0, 0, 0, 3, 0, 0, 1, 0, 0, 5, 4, 0, 0,
                            2, 7, 0, 0, 0, 1, 2, 0, 3, 0, 0, 0, 0, 7, 3, 0, 1, 0, 0, 0, 0, 0, 1, 0};

  std::vector<int> res(V, 0);

  std::vector<int> expected_res = {0, 2, 5, 5, 6, 9, 10};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(expected_res, res);
}
TEST(kapustin_dijkstras_algorithm, test_empty_graph) {
  const int V = 4;
  const int E = 0;

  std::vector<int> graph = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int> res(V, 0);
  std::vector<int> expected_res = {0, INT_MAX, INT_MAX, INT_MAX};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_res, res);
}

TEST(kapustin_dijkstras_algorithm, test_single_edge_graph) {
  const int V = 2;
  const int E = 1;

  std::vector<int> graph = {0, 5, 5, 0};

  std::vector<int> res(V, 0);
  std::vector<int> expected_res = {0, 5};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_res, res);
}

TEST(kapustin_dijkstras_algorithm, test_cyclic_graph) {
  const int V = 4;
  const int E = 4;

  std::vector<int> graph = {0, 1, 0, 10, 1, 0, 2, 0, 0, 2, 0, 3, 10, 0, 3, 0};

  std::vector<int> res(V, 0);
  std::vector<int> expected_res = {0, 1, 3, 6};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_res, res);
}
TEST(kapustin_dijkstras_algorithm, test_disconnected_graph) {
  const int V = 4;
  const int E = 2;

  std::vector<int> graph = {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0};

  std::vector<int> res(V, 0);
  std::vector<int> expected_res = {0, 1, INT_MAX, INT_MAX};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_res, res);
}

TEST(kapustin_dijkstras_algorithm, test_large_weight_range_graph) {
  const int V = 4;
  const int E = 3;

  std::vector<int> graph = {0, 1, 1000, 0, 1, 0, 10, 0, 1000, 10, 0, 5, 0, 0, 5, 0};

  std::vector<int> res(V, 0);
  std::vector<int> expected_res = {0, 1, 11, 16};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_res, res);
}