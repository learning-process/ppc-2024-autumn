// Copyright 2024 Nesterov Alexander
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/gusev_n_dijkstras_algorithm/include/ops_seq.hpp"

TEST(gusev_n_dijkstras_algorithm_seq, test_pipeline_run) {
  const int num_vertices = 1000;
  gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential::SparseGraphCRS graph;
  graph.num_vertices = num_vertices;

  graph.row_ptr.resize(num_vertices + 1);
  graph.col_indices.reserve(num_vertices * 10);
  graph.values.reserve(num_vertices * 10);

  for (int i = 0; i < num_vertices; ++i) {
    graph.row_ptr[i] = graph.col_indices.size();
    for (int j = 1; j <= 10; ++j) {
      int neighbor = (i + j) % num_vertices;
      graph.col_indices.push_back(neighbor);
      graph.values.push_back(static_cast<double>(rand() % 10 + 1));
    }
  }
  graph.row_ptr[num_vertices] = graph.col_indices.size();

  std::vector<double> distances(num_vertices);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&graph));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));

  auto testTask = std::make_shared<gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential>(taskData);

  ASSERT_TRUE(testTask->validation());
  ASSERT_TRUE(testTask->pre_processing());
  ASSERT_TRUE(testTask->run());
  ASSERT_TRUE(testTask->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(gusev_n_dijkstras_algorithm_seq, test_task_run) {
  const int num_vertices = 10000;
  gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential::SparseGraphCRS graph;
  graph.num_vertices = num_vertices;

  graph.row_ptr.resize(num_vertices + 1);
  graph.col_indices.reserve(num_vertices);
  graph.values.reserve(num_vertices);

  for (int i = 0; i < num_vertices; ++i) {
    graph.row_ptr[i] = graph.col_indices.size();
    if (i % 100 == 0 && i + 1 < num_vertices) {
      graph.col_indices.push_back(i + 1);
      graph.values.push_back(1.0);
    }
  }
  graph.row_ptr[num_vertices] = graph.col_indices.size();

  std::vector<double> distances(num_vertices);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&graph));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));

  auto testTask = std::make_shared<gusev_n_dijkstras_algorithm_seq::DijkstrasAlgorithmSequential>(taskData);

  ASSERT_TRUE(testTask->validation());
  ASSERT_TRUE(testTask->pre_processing());
  ASSERT_TRUE(testTask->run());
  ASSERT_TRUE(testTask->post_processing());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);

  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
