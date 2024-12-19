// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/makhov_m_ring_topology/include/ops_mpi.hpp"

TEST(mpi_makhov_m_ring_topology_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  size_t size = 10000000;
  std::vector<int32_t> input_vector(size, 1);
  std::vector<int32_t> output_vector(size);
  std::vector<int32_t> sequence(world.size() + 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequence.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.size() + 1);
  }

  auto testMpiTaskParallel = std::make_shared<makhov_m_ring_topology::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(mpi_makhov_m_ring_topology_perf_test, test_task_run) {
  boost::mpi::communicator world;
  size_t size = 10000000;
  std::vector<int32_t> input_vector(size, 1);
  std::vector<int32_t> output_vector(size);
  std::vector<int32_t> sequence(world.size() + 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequence.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.size() + 1);
  }

  auto testMpiTaskParallel = std::make_shared<makhov_m_ring_topology::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
