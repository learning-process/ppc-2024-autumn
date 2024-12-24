#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/zinoviev_a_bellman_ford/include/ops_mpi.hpp"

namespace zinoviev_a_bellman_ford_mpi {

std::vector<int> generateRandomGraph(size_t V, size_t E) {
  std::vector<int> graph(V * V, 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> weightDist(1, 100);
  std::uniform_int_distribution<> vertexDist(0, V - 1);

  size_t edges = 0;
  while (edges < E) {
    int u = vertexDist(gen);
    int v = vertexDist(gen);
    if (u != v && graph[u * V + v] == 0) {
      graph[u * V + v] = weightDist(gen);
      edges++;
    }
  }

  return graph;
}

}  // namespace zinoviev_a_bellman_ford_mpi

TEST(zinoviev_a_bellman_ford, test_pipeline_run) {
  boost::mpi::communicator world;
  size_t V = 1000;
  size_t E = 5000;
  std::vector<int> graph = zinoviev_a_bellman_ford_mpi::generateRandomGraph(V, E);
  std::vector<int> shortest_paths(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskData->inputs_count.emplace_back(V);
  taskData->inputs_count.emplace_back(E);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(shortest_paths.data()));
  taskData->outputs_count.emplace_back(V);

  auto task = std::make_shared<zinoviev_a_bellman_ford_mpi::BellmanFordMPIMPI>(taskData);
  ASSERT_EQ(task->validation(), true);
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(zinoviev_a_bellman_ford, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> graph = {0, 4, 0, 0, 0, 0,  0, 8, 0, 4, 0,  8, 0, 0, 0,  0, 11, 0, 0, 8, 0, 7,  0,  4, 0, 0, 2,
                            0, 0, 7, 0, 9, 14, 0, 0, 0, 0, 0,  0, 9, 0, 10, 0, 0,  0, 0, 0, 4, 14, 10, 0, 2, 0, 0,
                            0, 0, 0, 0, 0, 2,  0, 1, 6, 8, 11, 0, 0, 0, 0,  1, 0,  7, 0, 0, 2, 0,  0,  0, 6, 7, 0};
  std::vector<int> shortest_paths(9, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskData->inputs_count.emplace_back(9);
  taskData->inputs_count.emplace_back(28);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(shortest_paths.data()));
  taskData->outputs_count.emplace_back(9);

  auto task = std::make_shared<zinoviev_a_bellman_ford_mpi::BellmanFordMPIMPI>(taskData);
  ASSERT_EQ(task->validation(), true);
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}