#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vavilov_v_bellman_ford/include/ops_mpi.hpp"

std::vector<std::tuple<int, int, int>> generate_linear_graph(int num_vertices) {
    std::vector<std::tuple<int, int, int>> edges;
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

TEST(vavilov_v_bellman_ford_mpi, test_task_run) {
  boost::mpi::communicator world;

  const int num_vertices = 1000;
  auto edges = generate_linear_graph(num_vertices);
  auto expected_distances = compute_expected_distances(num_vertices);

  std::vector<int> distances(num_vertices, INT_MAX);
  distances[0] = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
    taskDataPar->inputs_count.emplace_back(edges.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
    taskDataPar->outputs_count.emplace_back(distances.size());
  }

  auto testMpiTaskParallel = std::make_shared<vavilov_v_bellman_ford_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] {return current_timer.elapsed();};

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(distances, expected_distances);
  }
}

TEST(vavilov_v_bellman_ford_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int num_vertices = 1000;
  auto edges = generate_linear_graph(num_vertices);
  auto expected_distances = compute_expected_distances(num_vertices);

  std::vector<int> distances(num_vertices, INT_MAX);
  distances[0] = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(edges.data()));
    taskDataPar->inputs_count.emplace_back(edges.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
    taskDataPar->outputs_count.emplace_back(distances.size());
  }

  auto testMpiTaskParallel = std::make_shared<vavilov_v_bellman_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] {return current_timer.elapsed();};

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(distances, expected_distances);
  }
}
