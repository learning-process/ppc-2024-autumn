#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "mpi/kapustin_dijk_alg_mpi/include/mpi_alg.hpp"
namespace generateGraph {
std::vector<int> generateGraph(int V, int E) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> graph(V * V, 0);
  std::set<std::pair<int, int>> edges;

  std::uniform_int_distribution<> dist_vertex(0, V - 1);
  std::uniform_int_distribution<> dist_weight(1, 100);

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

TEST(kapustin_dijkstras_algorithm_mpi, test_pipeline_run) {
  const int V = 7000;
  const int E = 14000;
  boost::mpi::communicator world;
  std::vector<int> graph = generateGraph::generateGraph(V, E);
  std::vector<int> resMPI(V, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataMPI->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataMPI->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
  taskDataMPI->outputs_count.emplace_back(resMPI.size());
  auto testTaskMPI = std::make_shared<kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI>(taskDataMPI);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  ppc::core::Perf perfAnalyzer(testTaskMPI);
  perfAnalyzer.pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(kapustin_dijkstras_algorithm_mpi, test_task_run) {
  const int V = 7000;
  const int E = 14000;
  boost::mpi::communicator world;
  std::vector<int> graph = generateGraph::generateGraph(V, E);
  std::vector<int> resMPI(V, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph.data()));
  taskDataMPI->inputs_count.emplace_back(static_cast<uint32_t>(V));
  taskDataMPI->inputs_count.emplace_back(static_cast<uint32_t>(E));
  taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t*>(resMPI.data()));
  taskDataMPI->outputs_count.emplace_back(resMPI.size());
  auto testTaskMPI = std::make_shared<kapustin_dijkstras_algorithm_mpi::DijkstrasAlgorithmMPI>(taskDataMPI);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  ppc::core::Perf perfAnalyzer(testTaskMPI);
  perfAnalyzer.task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}