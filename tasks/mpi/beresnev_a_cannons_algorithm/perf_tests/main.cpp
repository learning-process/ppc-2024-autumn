// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/beresnev_a_cannons_algorithm/include/ops_mpi.hpp"

TEST(beresnev_a_cannons_algorithm_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int n = 500;
  std::vector<double> inA;
  std::vector<double> inB;
  std::vector<double> outC;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inA = std::vector<double>(n * n, 0.0);
    inB = std::vector<double>(n * n, 1.0);
    outC = std::vector<double>(n * n);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size());
  }

  auto testMpiTaskParallel = std::make_shared<beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
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
    ASSERT_TRUE(
        std::equal(inA.begin(), inA.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
  }
}

TEST(beresnev_a_cannons_algorithm_mpi, test_task_run) {
  boost::mpi::communicator world;
  int n = 500;
  std::vector<double> inA;
  std::vector<double> inB;
  std::vector<double> outC;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inA = std::vector<double>(n * n, 0.0);
    inB = std::vector<double>(n * n, 1.0);
    outC = std::vector<double>(n * n, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size());
  }

  auto testMpiTaskParallel = std::make_shared<beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
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
    ASSERT_TRUE(
        std::equal(inA.begin(), inA.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
  }
}