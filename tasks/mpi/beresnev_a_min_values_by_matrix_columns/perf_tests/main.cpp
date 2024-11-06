// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/beresnev_a_min_values_by_matrix_columns/include/ops_mpi.hpp"

TEST(beresnev_a_min_values_by_matrix_columns_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int N = 1000;
  const int M = 1000;

  std::vector<int> in;
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);
  std::vector<int> gold(M, 0);
  gold[0] = -100;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = std::vector<int>(N * M, 0);
    in[0] = -100;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(out, gold);
  }
}

TEST(beresnev_a_min_values_by_matrix_columns_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int N = 1000;
  const int M = 1000;

  std::vector<int> in;
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);
  std::vector<int> gold(M, 0);
  gold[0] = -100;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = std::vector<int>(N * M, 0);
    in[0] = -100;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(out, gold);
  }
}