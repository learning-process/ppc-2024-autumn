#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/zinoviev_a_bellman_ford/include/ops_mpi.hpp"

TEST(zinoviev_a_bellman_ford, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_graph;
  std::vector<int> global_distances;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Example graph in CRS format
    global_graph = {0, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    global_distances = std::vector<int>(6, INT_MAX);
    global_distances[0] = 0;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
    taskDataPar->inputs_count.emplace_back(global_graph.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_distances.data()));
    taskDataPar->outputs_count.emplace_back(global_distances.size());
  }

  auto bellmanFordMPITaskParallel =
      std::make_shared<zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel>(taskDataPar);
  ASSERT_EQ(bellmanFordMPITaskParallel->validation(), true);
  bellmanFordMPITaskParallel->pre_processing();
  bellmanFordMPITaskParallel->run();
  bellmanFordMPITaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(bellmanFordMPITaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(zinoviev_a_bellman_ford, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_graph;
  std::vector<int> global_distances;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Example graph in CRS format
    global_graph = {0, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    global_distances = std::vector<int>(6, INT_MAX);
    global_distances[0] = 0;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
    taskDataPar->inputs_count.emplace_back(global_graph.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_distances.data()));
    taskDataPar->outputs_count.emplace_back(global_distances.size());
  }

  auto bellmanFordMPITaskParallel =
      std::make_shared<zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel>(taskDataPar);
  ASSERT_EQ(bellmanFordMPITaskParallel->validation(), true);
  bellmanFordMPITaskParallel->pre_processing();
  bellmanFordMPITaskParallel->run();
  bellmanFordMPITaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(bellmanFordMPITaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}