// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/morozov_e_min_val_in_rows_matrix/include/ops_mpi.hpp"

TEST(morozov_e_min_val_in_rows_matrix_perf_test, test_pipeline_run_my) {
  boost::mpi::communicator world;
  const int n = 5000;
  const int m = 5000;
  std::vector<std::vector<int>> matrix(n, std::vector<int>(m));
  std::vector<int> res(n);
  std::vector<int> res_ = morozov_e_min_val_in_rows_matrix::minValInRowsMatrix(matrix);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(n);
  }
  auto testMpiTaskParallel = std::make_shared<morozov_e_min_val_in_rows_matrix::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(res_, res);
  }
}
TEST(morozov_e_min_val_in_rows_matrix_perf_test, test_task_run_my) {
  boost::mpi::communicator world;
  const int n = 4500;
  const int m = 4500;
  std::vector<std::vector<int>> matrix(n, std::vector<int>(m));
  std::vector<int> res(n);
  std::vector<int> res_ = morozov_e_min_val_in_rows_matrix::minValInRowsMatrix(matrix);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (int i = 0; i < n; ++i) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(n);
  }
  auto testMpiTaskParallel = std::make_shared<morozov_e_min_val_in_rows_matrix::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(res_, res);
  }
}