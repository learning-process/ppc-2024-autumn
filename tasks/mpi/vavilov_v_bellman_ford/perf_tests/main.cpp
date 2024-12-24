#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vavilov_v_bellman_ford/include/ops_mpi.hpp"

std::vector<int> generate_linear_graph(int num_vertices) {
  std::vector<int> matrix(num_vertices * num_vertices, 0);
  for (int i = 0; i < num_vertices - 1; ++i) {
    matrix[i * num_vertices + i + 1] = 1;
  }
  return matrix;
}

std::vector<int> compute_expected_distances(int num_vertices) {
  std::vector<int> distances(num_vertices);
  for (int i = 0; i < num_vertices; ++i) {
    distances[i] = i;
  }
  return distances;
}

TEST(vavilov_v_bellman_ford_mpi, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const int num_vertices = 5000;
  const int edges_count = 4999;
  const int source = 0;

  auto matrix = generate_linear_graph(num_vertices);
  auto expected_distances = compute_expected_distances(num_vertices);

  std::vector<int> distances(num_vertices);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(num_vertices);
    taskDataPar->inputs_count.emplace_back(edges_count);
    taskDataPar->inputs_count.emplace_back(source);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->outputs_count.emplace_back(distances.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  }

  auto testMpiTaskParallel = std::make_shared<vavilov_v_bellman_ford_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(distances, expected_distances);
  }
}

TEST(vavilov_v_bellman_ford_mpi, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  const int num_vertices = 5000;
  const int edges_count = 4999;
  const int source = 0;

  auto matrix = generate_linear_graph(num_vertices);
  auto expected_distances = compute_expected_distances(num_vertices);

  std::vector<int> distances(num_vertices);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(num_vertices);
    taskDataPar->inputs_count.emplace_back(edges_count);
    taskDataPar->inputs_count.emplace_back(source);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->outputs_count.emplace_back(distances.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  }

  auto testMpiTaskParallel = std::make_shared<vavilov_v_bellman_ford_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(distances, expected_distances);
  }
}
