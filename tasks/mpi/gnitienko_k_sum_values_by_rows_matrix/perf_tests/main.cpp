// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gnitienko_k_sum_values_by_rows_matrix/include/ops_mpi.hpp"

TEST(gnitienko_k_sum_by_row_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_sums;
  std::vector<int> expect;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rows;
  int cols;
  if (world.rank() == 0) {
    rows = 10000;
    cols = 10000;
    expect.resize(rows, 10000);
    global_matrix.resize(rows * cols, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(rows));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(cols));
    global_sums.resize(rows, 0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sums.data()));
    taskDataPar->outputs_count.emplace_back(static_cast<uint32_t>(global_sums.size()));
  }

  auto sumByRowTask = std::make_shared<gnitienko_k_sum_row_mpi::SumByRowMPIParallel>(taskDataPar);
  ASSERT_EQ(sumByRowTask->validation(), true);
  sumByRowTask->pre_processing();
  sumByRowTask->run();
  sumByRowTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sumByRowTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(global_sums, expect);
  }
}

TEST(gnitienko_k_sum_by_row_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_sums;
  std::vector<int> expect;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rows;
  int cols;
  if (world.rank() == 0) {
    rows = 10000;
    cols = 10000;
    expect.resize(rows, 10000);
    global_matrix.resize(rows * cols, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(rows));
    taskDataPar->inputs_count.emplace_back(static_cast<uint32_t>(cols));
    global_sums.resize(rows, 0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sums.data()));
    taskDataPar->outputs_count.emplace_back(static_cast<uint32_t>(global_sums.size()));
  }

  auto sumByRowTask = std::make_shared<gnitienko_k_sum_row_mpi::SumByRowMPIParallel>(taskDataPar);
  ASSERT_EQ(sumByRowTask->validation(), true);
  sumByRowTask->pre_processing();
  sumByRowTask->run();
  sumByRowTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sumByRowTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(global_sums, expect);
  }
}