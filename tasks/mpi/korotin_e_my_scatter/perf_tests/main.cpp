// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/korotin_e_my_scatter/include/ops_mpi.hpp"

TEST(korotin_e_my_scatter, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> matrix;
  std::vector<double> max_val(1, 0);
  int root = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rows;
  int columns;
  if (world.rank() == root) {
    rows = columns = 1000;
    matrix = std::vector<double>(rows * columns, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_val.data()));
    taskDataPar->outputs_count.emplace_back(max_val.size());
  }

  auto testMpiTaskParallel = std::make_shared<korotin_e_my_scatter_mpi::TestMPITaskMyParallel>(taskDataPar, root);
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
  if (world.rank() == root) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_DOUBLE_EQ(1, max_val[0]);
  }
}

TEST(korotin_e_my_scatter, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> matrix;
  std::vector<double> max_val(1, 0);
  int root = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rows;
  int columns;
  if (world.rank() == root) {
    rows = columns = 1000;
    matrix = std::vector<double>(rows * columns, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_val.data()));
    taskDataPar->outputs_count.emplace_back(max_val.size());
  }

  auto testMpiTaskParallel = std::make_shared<korotin_e_my_scatter_mpi::TestMPITaskMyParallel>(taskDataPar, root);
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
  if (world.rank() == root) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_DOUBLE_EQ(1, max_val[0]);
  }
}
