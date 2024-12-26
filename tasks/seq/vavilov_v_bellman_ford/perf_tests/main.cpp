#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vavilov_v_bellman_ford/include/ops_seq.hpp"

std::vector<int> generate_linear_graph(int num_vertices) {
  std::vector<int> edges((num_vertices - 1) * 3);
  for (int i = 0; i < num_vertices - 1; ++i) {
    edges[i * 3] = i;
    edges[i * 3 + 1] = i + 1;
    edges[i * 3 + 2] = 1;
  }
  return edges;
}

std::vector<int> compute_expected_distances(int num_vertices) {
  std::vector<int> distances(num_vertices);
  for (int i = 0; i < num_vertices; ++i) {
    distances[i] = i;
  }
  return distances;
}

TEST(vavilov_v_bellman_ford_seq, test_task_run) {
  const int num_vertices = 1000;
  const int edges_count = 999;
  const int source = 0;
  auto edges = generate_linear_graph(num_vertices);
  auto expected_distances = compute_expected_distances(num_vertices);

  std::vector<int> distances(num_vertices);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(num_vertices);
  taskDataSeq->inputs_count.emplace_back(edges_count);
  taskDataSeq->inputs_count.emplace_back(source);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
  taskDataSeq->outputs_count.emplace_back(distances.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));

  auto testTaskSequential = std::make_shared<vavilov_v_bellman_ford_seq::TestTaskSequential>(taskDataSeq);

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

  ASSERT_EQ(distances, expected_distances);
}

TEST(vavilov_v_bellman_ford_seq, test_pipeline_run) {
  const int num_vertices = 1000;
  const int edges_count = 999;
  const int source = 0;
  auto edges = generate_linear_graph(num_vertices);
  auto expected_distances = compute_expected_distances(num_vertices);

  std::vector<int> distances(num_vertices);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(num_vertices);
  taskDataSeq->inputs_count.emplace_back(edges_count);
  taskDataSeq->inputs_count.emplace_back(source);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
  taskDataSeq->outputs_count.emplace_back(distances.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));

  auto testTaskSequential = std::make_shared<vavilov_v_bellman_ford_seq::TestTaskSequential>(taskDataSeq);

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

  ASSERT_EQ(distances, expected_distances);
}
