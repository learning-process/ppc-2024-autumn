// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/grudzin_k_all_reduce/include/ops_mpi.hpp"

TEST(grudzin_k_all_reduce_mpi, test_pipeline_run) {
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
  auto testMpiTaskMyRealization = std::make_shared<grudzin_k_all_reduce_mpi::TestMPITaskMyRealization>(taskDataPar);
  ASSERT_EQ(testMpiTaskMyRealization->validation(), true);
  testMpiTaskMyRealization->pre_processing();
  testMpiTaskMyRealization->run();
  testMpiTaskMyRealization->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 20;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskMyRealization);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(parallel_results, global_ans);
  }
}

TEST(grudzin_k_all_reduce_mpi, test_task_run) {
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
  auto testMpiTaskMyRealization = std::make_shared<grudzin_k_all_reduce_mpi::TestMPITaskMyRealization>(taskDataPar);
  ASSERT_EQ(testMpiTaskMyRealization->validation(), true);
  testMpiTaskMyRealization->pre_processing();
  testMpiTaskMyRealization->run();
  testMpiTaskMyRealization->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 20;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskMyRealization);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(parallel_results, global_ans);
  }
}

// I use it in local nachine to analize speed,idk shoud  delete it or not so i keep it
// TEST(grudzin_k_all_reduce_mpi, test_pipeline_run_boost) {
//  int rows = 5000;
//  int colums = 5000;
//
//  boost::mpi::communicator world;
//  std::vector<int> mas(colums * rows, -1);
//  std::vector<int> parallel_results(colums);
//  std::vector<int> global_ans(colums, 5000);
//
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
//    taskDataPar->inputs_count.emplace_back(rows);
//    taskDataPar->inputs_count.emplace_back(colums);
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_results.data()));
//    taskDataPar->outputs_count.emplace_back(parallel_results.size());
//  }
//  auto testMpiTaskMyRealization =
//  std::make_shared<grudzin_k_all_reduce_mpi::TestMPITaskBoostRealization>(taskDataPar);
//  ASSERT_EQ(testMpiTaskMyRealization->validation(), true);
//  testMpiTaskMyRealization->pre_processing();
//  testMpiTaskMyRealization->run();
//  testMpiTaskMyRealization->post_processing();
//
//  // Create Perf attributes
//  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//  perfAttr->num_running = 20;
//  const boost::mpi::timer current_timer;
//  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
//
//  // Create and init perf results
//  auto perfResults = std::make_shared<ppc::core::PerfResults>();
//
//  // Create Perf analyzer
//  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskMyRealization);
//  perfAnalyzer->pipeline_run(perfAttr, perfResults);
//  if (world.rank() == 0) {
//    ppc::core::Perf::print_perf_statistic(perfResults);
//    ASSERT_EQ(parallel_results, global_ans);
//  }
//}
//
// TEST(grudzin_k_all_reduce_mpi, test_task_run_boost) {
//  int rows = 5000;
//  int colums = 5000;
//
//  boost::mpi::communicator world;
//  std::vector<int> mas(colums * rows, -1);
//  std::vector<int> parallel_results(colums);
//  std::vector<int> global_ans(colums, 5000);
//
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
//    taskDataPar->inputs_count.emplace_back(rows);
//    taskDataPar->inputs_count.emplace_back(colums);
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_results.data()));
//    taskDataPar->outputs_count.emplace_back(parallel_results.size());
//  }
//  auto testMpiTaskMyRealization =
//  std::make_shared<grudzin_k_all_reduce_mpi::TestMPITaskBoostRealization>(taskDataPar);
//  ASSERT_EQ(testMpiTaskMyRealization->validation(), true);
//  testMpiTaskMyRealization->pre_processing();
//  testMpiTaskMyRealization->run();
//  testMpiTaskMyRealization->post_processing();
//
//  // Create Perf attributes
//  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
//  perfAttr->num_running = 20;
//  const boost::mpi::timer current_timer;
//  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
//
//  // Create and init perf results
//  auto perfResults = std::make_shared<ppc::core::PerfResults>();
//
//  // Create Perf analyzer
//  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskMyRealization);
//  perfAnalyzer->pipeline_run(perfAttr, perfResults);
//  if (world.rank() == 0) {
//    ppc::core::Perf::print_perf_statistic(perfResults);
//    ASSERT_EQ(parallel_results, global_ans);
//  }
//}