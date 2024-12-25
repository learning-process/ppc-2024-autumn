// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gorbunov_e_check_lexicographic/include/ops_mpi.hpp"

TEST(gorbunov_e_check_lexicographic_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<char> global_word_left = std::vector<char>(10000000, 'A');
  std::vector<char> global_word_right = std::vector<char>(9999999, 'A');
  global_word_right.insert('B', 5000000);
  std::vector<int32_t> parallel_result(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_left.data()));
    taskDataPar->inputs_count.emplace_back(global_word_left.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_right.data()));
    taskDataPar->inputs_count.emplace_back(global_word_right.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_result.data()));
    taskDataPar->outputs_count.emplace_back(parallel_result.size());
  }

  auto testMpiTaskParallel = std::make_shared<gorbunov_e_check_lexicographic_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(1, parallel_result[0]);
  }
}


TEST(gorbunov_e_check_lexicographic_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<char> global_word_left = std::vector<char>(10000000, 'A');
  std::vector<char> global_word_right = std::vector<char>(9999999, 'A');
  global_word_right.insert('B', 5000000);
  std::vector<int32_t> parallel_result(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_left.data()));
    taskDataPar->inputs_count.emplace_back(global_word_left.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_right.data()));
    taskDataPar->inputs_count.emplace_back(global_word_right.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_result.data()));
    taskDataPar->outputs_count.emplace_back(parallel_result.size());
  }

  auto testMpiTaskParallel = std::make_shared<gorbunov_e_check_lexicographic_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(1, parallel_result[0]);
  }
}