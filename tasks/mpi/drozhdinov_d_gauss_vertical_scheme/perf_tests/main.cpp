// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
// not example
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/drozhdinov_d_gauss_vertical_scheme/include/ops_mpi.hpp"

TEST(MPIGAUSSPERF, test_pipeline_run) {
  boost::mpi::communicator world;
  int rows = 1000;
  int columns = 1000;
  std::vector<double> matrix = genElementaryMatrix(rows, columns);
  std::vector<double> b(rows * columns, 1);
  std::vector<double> expres_par(rows);
  std::vector<double> res(rows, 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(expres_par, res);
  }
}

TEST(MPIGAUSSPERF, test_task_run) {
  boost::mpi::communicator world;
  int rows = 1000;
  int columns = 1000;
  std::vector<double> matrix = genElementaryMatrix(rows, columns);
  std::vector<double> b(rows * columns, 1);
  std::vector<double> expres_par(rows);
  std::vector<double> res(rows, 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(expres_par, res);
  }
}
