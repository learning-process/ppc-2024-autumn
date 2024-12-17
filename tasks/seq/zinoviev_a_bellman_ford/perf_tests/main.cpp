// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/zinoviev_a_bellman_ford/include/ops_seq.hpp"

TEST(zinoviev_a_bellman_ford, test_pipeline_run) {
  const int num_vertices = 100;
  const int num_edges = 500;
  std::vector<int> graph = generateRandomGraph(num_vertices, num_edges);
  std::vector<int> dist(num_vertices, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(graph.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(dist.data()));
  taskDataSeq->outputs_count.emplace_back(dist.size());

  // Create Task
  auto testSeqTaskSequential = std::make_shared<zinoviev_a_bellman_ford_seq::BellmanFordSeqTaskSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSeqTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(zinoviev_a_bellman_ford, test_task_run) {
  const int num_vertices = 100;
  const int num_edges = 500;
  std::vector<int> graph = generateRandomGraph(num_vertices, num_edges);
  std::vector<int> dist(num_vertices, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataSeq->inputs_count.emplace_back(graph.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(dist.data()));
  taskDataSeq->outputs_count.emplace_back(dist.size());

  // Create Task
  auto testSeqTaskSequential = std::make_shared<zinoviev_a_bellman_ford_seq::BellmanFordSeqTaskSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSeqTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}