#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gusev_n_hypercube_topology/include/ops_mpi.hpp"

TEST(gusev_n_hypercube_topology_mpi, run_pipeline) {
  boost::mpi::communicator world;

  std::vector<uint8_t> input_data(1024, 1);
  std::vector<uint8_t> output_data(1024);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data.data());
  task_data->inputs_count.push_back(input_data.size());
  task_data->outputs.push_back(output_data.data());
  task_data->outputs_count.push_back(output_data.size());

  gusev_n_hypercube_topology_mpi::HypercubeTopologyParallel task(task_data);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  perfAttr->num_running = 100;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  ppc::core::Perf perfAnalyzer(std::make_shared<gusev_n_hypercube_topology_mpi::HypercubeTopologyParallel>(task_data));
  perfAnalyzer.pipeline_run(perfAttr, perfResults);
  // for (uint64_t i = 0; i < perfAttr->num_running; ++i) {
  //   perfAnalyzer.pipeline_run(perfAttr, perfResults);
  // }
  if (world.rank() == 0) ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(gusev_n_hypercube_topology_mpi, run_task) {
  boost::mpi::communicator world;

  std::vector<uint8_t> input_data(1024, 1);
  std::vector<uint8_t> output_data(1024);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(input_data.data());
  task_data->inputs_count.push_back(input_data.size());
  task_data->outputs.push_back(output_data.data());
  task_data->outputs_count.push_back(output_data.size());

  gusev_n_hypercube_topology_mpi::HypercubeTopologyParallel task(task_data);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  perfAttr->num_running = 100;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  ppc::core::Perf perfAnalyzer(std::make_shared<gusev_n_hypercube_topology_mpi::HypercubeTopologyParallel>(task_data));
  perfAnalyzer.task_run(perfAttr, perfResults);
  if (world.rank() == 0) ppc::core::Perf::print_perf_statistic(perfResults);
}