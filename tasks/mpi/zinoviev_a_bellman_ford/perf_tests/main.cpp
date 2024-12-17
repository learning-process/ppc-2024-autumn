// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/zinoviev_a_bellman_ford/include/ops_mpi.hpp"

TEST(zinoviev_a_bellman_ford, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_graph;
  std::vector<int> global_dist(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_graph;
  if (world.rank() == 0) {
    count_size_graph = 120;
    global_graph = std::vector<int>(count_size_graph, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
    taskDataPar->inputs_count.emplace_back(global_graph.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_dist.data()));
    taskDataPar->outputs_count.emplace_back(global_dist.size());
  }

  auto bellmanFordMpiTaskParallel =
      std::make_shared<zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel>(taskDataPar);
  ASSERT_EQ(bellmanFordMpiTaskParallel->validation(), true);
  bellmanFordMpiTaskParallel->pre_processing();
  bellmanFordMpiTaskParallel->run();
  bellmanFordMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(bellmanFordMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(zinoviev_a_bellman_ford, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_graph;
  std::vector<int> global_dist(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_graph;
  if (world.rank() == 0) {
    count_size_graph = 120;
    global_graph = std::vector<int>(count_size_graph, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_graph.data()));
    taskDataPar->inputs_count.emplace_back(global_graph.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_dist.data()));
    taskDataPar->outputs_count.emplace_back(global_dist.size());
  }

  auto bellmanFordMpiTaskParallel =
      std::make_shared<zinoviev_a_bellman_ford_mpi::BellmanFordMPITaskParallel>(taskDataPar);
  ASSERT_EQ(bellmanFordMpiTaskParallel->validation(), true);
  bellmanFordMpiTaskParallel->pre_processing();
  bellmanFordMpiTaskParallel->run();
  bellmanFordMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(bellmanFordMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}