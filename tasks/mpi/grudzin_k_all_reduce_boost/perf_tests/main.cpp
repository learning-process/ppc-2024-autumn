// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/grudzin_k_all_reduce_boost/include/ops_mpi.hpp"

TEST(grudzin_k_all_reduce_boost_mpi, test_pipeline_run) {
  int rows = 5000;
  int colums = 5000;

  boost::mpi::communicator world;
  std::vector<int> mas(colums * rows, -1);
  std::vector<int> parallel_results(colums);
  std::vector<int> global_ans(colums, 5000);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(colums);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_results.data()));
    taskDataPar->outputs_count.emplace_back(parallel_results.size());
  }
  auto TestMPITaskBoostRealization =
      std::make_shared<grudzin_k_all_reduce_boost_mpi::TestMPITaskBoostRealization>(taskDataPar);
  ASSERT_EQ(TestMPITaskBoostRealization->validation(), true);
  TestMPITaskBoostRealization->pre_processing();
  TestMPITaskBoostRealization->run();
  TestMPITaskBoostRealization->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 20;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestMPITaskBoostRealization);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(parallel_results, global_ans);
  }
}

TEST(grudzin_k_all_reduce_boost_mpi, test_task_run) {
  int rows = 5000;
  int colums = 5000;

  boost::mpi::communicator world;
  std::vector<int> mas(colums * rows, -1);
  std::vector<int> parallel_results(colums);
  std::vector<int> global_ans(colums, 5000);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(colums);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_results.data()));
    taskDataPar->outputs_count.emplace_back(parallel_results.size());
  }
  auto TestMPITaskBoostRealization =
      std::make_shared<grudzin_k_all_reduce_boost_mpi::TestMPITaskBoostRealization>(taskDataPar);
  ASSERT_EQ(TestMPITaskBoostRealization->validation(), true);
  TestMPITaskBoostRealization->pre_processing();
  TestMPITaskBoostRealization->run();
  TestMPITaskBoostRealization->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 20;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestMPITaskBoostRealization);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(parallel_results, global_ans);
  }
}
