#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gusev_n_dijkstras_algorithm/include/ops_mpi.hpp"

TEST(gusev_n_dijkstras_algorithm_mpi, run_pipeline) {
  boost::mpi::communicator world;

  auto graph = std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS>(100);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> weight_dist(0.1, 10.0);
  std::uniform_int_distribution<> vertex_dist(0, 99);

  for (int i = 0; i < 1000; ++i) {
    int u = vertex_dist(gen);
    int v = vertex_dist(gen);
    double weight = weight_dist(gen);
    if (u != v) {
      graph->add_edge(u, v, weight);
    }
  }

  std::vector<double> output_data(100);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(graph.get()));
  task_data->inputs_count.push_back(
      sizeof(gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size() * sizeof(double));

  gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel task(task_data);
  ASSERT_TRUE(task.validation());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  perfAttr->num_running = 5;

  ppc::core::Perf perfAnalyzer(
      std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel>(task_data));

  for (uint64_t i = 0; i < perfAttr->num_running; ++i) {
    perfAnalyzer.pipeline_run(perfAttr, perfResults);
  }

  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(gusev_n_dijkstras_algorithm_mpi, run_task) {
  boost::mpi::communicator world;

  auto graph = std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS>(100);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> weight_dist(0.1, 10.0);
  std::uniform_int_distribution<> vertex_dist(0, 99);

  for (int i = 0; i < 1000; ++i) {
    int u = vertex_dist(gen);
    int v = vertex_dist(gen);
    double weight = weight_dist(gen);
    if (u != v) {
      graph->add_edge(u, v, weight);
    }
  }

  std::vector<double> output_data(100);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(graph.get()));
  task_data->inputs_count.push_back(
      sizeof(gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel::SparseGraphCRS));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size() * sizeof(double));

  gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel task(task_data);
  ASSERT_TRUE(task.validation());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  perfAttr->num_running = 5;

  ppc::core::Perf perfAnalyzer(
      std::make_shared<gusev_n_dijkstras_algorithm_mpi::DijkstrasAlgorithmParallel>(task_data));

  for (uint64_t i = 0; i < perfAttr->num_running; ++i) {
    perfAnalyzer.task_run(perfAttr, perfResults);
  }

  ppc::core::Perf::print_perf_statistic(perfResults);
}
