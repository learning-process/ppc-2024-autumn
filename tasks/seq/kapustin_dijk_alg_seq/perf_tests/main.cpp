#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/kapustin_dijk_alg_seq/include/seq_alg.hpp"

namespace generateGraph {
const int INF = 1000000000;
std::vector<int> generateGraph(int V, int E) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> graph(V * V, 0);
  std::set<std::pair<int, int>> edges;
  std::uniform_int_distribution<> dist_vertex(0, V - 1);
  std::uniform_int_distribution<> dist_weight(1, 10);

  int edgeCount = 0;
  while (edgeCount < E) {
    int u = dist_vertex(gen);
    int v = dist_vertex(gen);
    if (u != v && edges.find({u, v}) == edges.end()) {
      int weight = dist_weight(gen);
      graph[u * V + v] = weight;
      graph[v * V + u] = weight;
      edges.insert({u, v});
      edges.insert({v, u});
      edgeCount++;
    }
  }
  return graph;
}
}  // namespace generateGraph

TEST(kapustin_dijkstras_algorithm, test_pipeline_run) {
  const int V = 10000;
  const int E = 700000;

  std::vector<int> graph = generateGraph::generateGraph(V, E);
  std::vector<int> res(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  auto testTaskSequential = std::make_shared<kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential>(taskDataSeq);

  std::vector<int> shortest_paths(V, generateGraph::INF);
  shortest_paths[0] = 0;

  std::set<int> processed;
  while (processed.size() < static_cast<size_t>(V)) {
    int u = -1;

    for (int i = 0; i < V; i++) {
      if (processed.find(i) == processed.end() && (u == -1 || shortest_paths[i] < shortest_paths[u])) {
        u = i;
      }
    }

    if (u == -1 || shortest_paths[u] == generateGraph::INF) {
      break;
    }

    processed.insert(u);

    for (int v = 0; v < V; v++) {
      if (graph[u * V + v] != 0 && processed.find(v) == processed.end()) {
        shortest_paths[v] = std::min(shortest_paths[v], shortest_paths[u] + graph[u * V + v]);
      }
    }
  }

  for (int i = 0; i < V; i++) {
    if (shortest_paths[i] == generateGraph::INF) {
      shortest_paths[i] = 0;
    }
  }

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(shortest_paths, res);
}

TEST(kapustin_dijkstras_algorithm, test_task_run) {
  const int V = 10000;
  const int E = 7000000;

  std::vector<int> graph = generateGraph::generateGraph(V, E);
  std::vector<int> res(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  auto testTaskSequential = std::make_shared<kapustin_i_dijkstra_algorithm::DijkstrasAlgorithmSequential>(taskDataSeq);

  std::vector<int> shortest_paths(V, generateGraph::INF);
  shortest_paths[0] = 0;

  std::set<int> processed;
  while (processed.size() < static_cast<size_t>(V)) {
    int u = -1;

    for (int i = 0; i < V; i++) {
      if (processed.find(i) == processed.end() && (u == -1 || shortest_paths[i] < shortest_paths[u])) {
        u = i;
      }
    }

    if (u == -1 || shortest_paths[u] == generateGraph::INF) {
      break;
    }

    processed.insert(u);

    for (int v = 0; v < V; v++) {
      if (graph[u * V + v] != 0 && processed.find(v) == processed.end()) {
        shortest_paths[v] = std::min(shortest_paths[v], shortest_paths[u] + graph[u * V + v]);
      }
    }
  }

  for (int i = 0; i < V; i++) {
    if (shortest_paths[i] == generateGraph::INF) {
      shortest_paths[i] = 0;
    }
  }

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(shortest_paths, res);
}