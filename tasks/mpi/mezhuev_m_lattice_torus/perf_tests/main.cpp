#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/mezhuev_m_lattice_torus/include/mpi.hpp"

TEST(mezhuev_m_lattice_torus, perf_test_pipeline_run) {
  boost::mpi::communicator world;

  int grid_size = std::sqrt(world.size());
  if (grid_size * grid_size != world.size() || world.size() < 4) {
    return;
  }

  const int count = 2020;
  std::vector<uint8_t> input_data(count, 1);
  std::vector<uint8_t> output_data(count);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(input_data.data());
  taskDataPar->inputs_count.emplace_back(input_data.size());
  taskDataPar->outputs.emplace_back(output_data.data());
  taskDataPar->outputs_count.emplace_back(output_data.size());

  auto gridTorusTask = std::make_shared<mezhuev_m_lattice_torus::GridTorusTopologyParallel>(taskDataPar);

  ASSERT_TRUE(gridTorusTask->validation());
  gridTorusTask->pre_processing();
  gridTorusTask->run();
  gridTorusTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gridTorusTask);

  for (uint64_t i = 0; i < perfAttr->num_running; ++i) {
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
  }

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(input_data, output_data);
  }
}

TEST(mezhuev_m_lattice_torus, perf_test_task_run) {
  boost::mpi::communicator world;

  int grid_size = std::sqrt(world.size());
  if (grid_size * grid_size != world.size() || world.size() < 4) {
    return;
  }

  const int count = 2020;
  std::vector<uint8_t> input_data(count, 1);
  std::vector<uint8_t> output_data(count);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(input_data.data());
  taskDataPar->inputs_count.emplace_back(input_data.size());
  taskDataPar->outputs.emplace_back(output_data.data());
  taskDataPar->outputs_count.emplace_back(output_data.size());

  auto gridTorusTask = std::make_shared<mezhuev_m_lattice_torus::GridTorusTopologyParallel>(taskDataPar);

  ASSERT_TRUE(gridTorusTask->validation());
  gridTorusTask->pre_processing();
  gridTorusTask->run();
  gridTorusTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(gridTorusTask);

  for (uint64_t i = 0; i < perfAttr->num_running; ++i) {
    perfAnalyzer->task_run(perfAttr, perfResults);
  }

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(input_data, output_data);
  }
}