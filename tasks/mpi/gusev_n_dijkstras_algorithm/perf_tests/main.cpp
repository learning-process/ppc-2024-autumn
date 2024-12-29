#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gusev_n_dijkstras_algorithm/include/ops_mpi.hpp"

void create_cycle_graph(
    std::shared_ptr<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS>& graph,
    int num_vertices) {
  for (int i = 0; i < num_vertices; ++i) {
    double weight = static_cast<double>(rand()) / RAND_MAX * 10.0 + 0.1;  // Random weight between 0.1 and 10.0
    graph->add_edge(i, (i + 1) % num_vertices, weight);                   // Connect in a cycle
  }
}

void create_bipartite_graph(
    std::shared_ptr<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS>& graph,
    int num_vertices) {
  int half = num_vertices / 2;
  for (int u = 0; u < half; ++u) {
    for (int v = half; v < num_vertices; ++v) {
      double weight = static_cast<double>(rand()) / RAND_MAX * 10.0 + 0.1;  // Random weight between 0.1 and 10.0
      graph->add_edge(u, v, weight);
    }
  }
}

TEST(gusev_n_dijkstras_algorithm_mpi, run_pipeline) {
  boost::mpi::communicator world;

  auto graph = std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS>(100);
  int num_vertices = 100;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> weight_dist(0.1, 10.0);
  std::uniform_int_distribution<> vertex_dist(0, 99);

  create_cycle_graph(graph, num_vertices);

  std::vector<double> output_data(100);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(graph.get()));
  task_data->inputs_count.push_back(
      sizeof(gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size() * sizeof(double));

  gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel task(task_data);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  ppc::core::Perf perfAnalyzer(
      std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel>(task_data));
  perfAnalyzer.pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(gusev_n_dijkstras_algorithm_mpi, run_task) {
  boost::mpi::communicator world;

  auto graph = std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS>(100);
  int num_vertices = 100;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> weight_dist(0.1, 10.0);
  std::uniform_int_distribution<> vertex_dist(0, 99);

  create_bipartite_graph(graph, num_vertices);

  std::vector<double> output_data(100);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(graph.get()));
  task_data->inputs_count.push_back(
      sizeof(gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size() * sizeof(double));

  gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel task(task_data);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  ppc::core::Perf perfAnalyzer(
      std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel>(task_data));
  perfAnalyzer.task_run(perfAttr, perfResults);

  if (world.rank() == 0) ppc::core::Perf::print_perf_statistic(perfResults);
}
