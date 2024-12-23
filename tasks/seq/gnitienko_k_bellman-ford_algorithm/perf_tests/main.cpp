// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/gnitienko_k_bellman-ford_algorithm/include/ops_seq.hpp"

namespace gnitienko_k_func {

const int INF = std::numeric_limits<int>::max();
const int MIN_WEIGHT = -10;
const int MAX_WEIGHT = 10;
std::vector<int> generateGraph(const int V) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dis(MIN_WEIGHT, MAX_WEIGHT);

  std::vector<int> graph(V * V, 0);

  for (int i = 0; i < V; ++i) {
    for (int j = i + 1; j < V; ++j) {
      int weight = dis(gen);
      graph[i * V + j] = weight;
    }
  }

  return graph;
}
}  // namespace gnitienko_k_func

TEST(gnitienko_k_bellman_ford_algorithm, test_pipeline_run) {
  const int V = 1000;
  const int E = 300000;

  // Create data
  std::vector<int> graph = gnitienko_k_func::generateGraph(V);
  std::vector<int> res(V, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSequential = std::make_shared<gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq>(taskDataSeq);

  std::vector<int> shortest_paths(V, gnitienko_k_func::INF);
  shortest_paths[0] = 0;

  bool changed = true;
  for (int k = 0; k < V - 1; ++k) {
    changed = false;
    for (int i = 0; i < V; ++i) {
      for (int j = 0; j < V; ++j) {
        int weight = graph[i * V + j];
        if (weight != 0) {
          if (shortest_paths[i] != gnitienko_k_func::INF && shortest_paths[i] + weight < shortest_paths[j]) {
            shortest_paths[j] = shortest_paths[i] + weight;
            changed = true;
          }
        }
      }
    }
    if (!changed) break;
  }

  for (int i = 0; i < V; i++) {
    if (shortest_paths[i] == gnitienko_k_func::INF) {
      shortest_paths[i] = 0;
    }
  }

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(shortest_paths, res);
}

TEST(gnitienko_k_bellman_ford_algorithm, test_task_run) {
  const int V = 10000;
  const int E = 3000000;

  // Create data
  std::vector<int> graph = gnitienko_k_func::generateGraph(V);
  std::vector<int> res(V, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  auto testTaskSequential = std::make_shared<gnitienko_k_bellman_ford_algorithm_seq::BellmanFordAlgSeq>(taskDataSeq);

  std::vector<int> shortest_paths(V, gnitienko_k_func::INF);
  shortest_paths[0] = 0;

  bool changed = true;
  for (int k = 0; k < V - 1; ++k) {
    changed = false;
    for (int i = 0; i < V; ++i) {
      for (int j = 0; j < V; ++j) {
        int weight = graph[i * V + j];
        if (weight != 0) {
          if (shortest_paths[i] != gnitienko_k_func::INF && shortest_paths[i] + weight < shortest_paths[j]) {
            shortest_paths[j] = shortest_paths[i] + weight;
            changed = true;
          }
        }
      }
    }
    if (!changed) break;
  }

  for (int i = 0; i < V; i++) {
    if (shortest_paths[i] == gnitienko_k_func::INF) {
      shortest_paths[i] = 0;
    }
  }

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(shortest_paths, res);
}