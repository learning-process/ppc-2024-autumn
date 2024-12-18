#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vavilov_v_bellman_ford/include/ops_seq.hpp"

std::vector<std::tuple<int, int, int>> generate_linear_graph(int num_vertices) {
  std::vector<std::tuple<int, int, int>> edges;
  edges.reserve(num_vertices - 1);
  for (int i = 0; i < num_vertices - 1; ++i) {
    edges.emplace_back(i, i + 1, i + 1);
  }
  return edges;
}

std::vector<int> compute_expected_distances(int num_vertices) {
  std::vector<int> distances(num_vertices, 0);
  for (int i = 1; i < num_vertices; ++i) {
    distances[i] = distances[i - 1] + i;
  }
  return distances;
}

TEST(vavilov_v_bellman_ford_seq, test_task_run) {
  const int num_vertices = 1000;
  auto edges = generate_linear_graph(num_vertices);
  auto expected_distances = compute_expected_distances(num_vertices);

  std::vector<int> distances(num_vertices, INT_MAX);
  distances[0] = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
  taskDataSeq->inputs_count.emplace_back(edges.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  taskDataSeq->outputs_count.emplace_back(distances.size());

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
  auto edges = generate_linear_graph(num_vertices);
  auto expected_distances = compute_expected_distances(num_vertices);

  std::vector<int> distances(num_vertices, INT_MAX);
  distances[0] = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
  taskDataSeq->inputs_count.emplace_back(edges.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  taskDataSeq->outputs_count.emplace_back(distances.size());

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
