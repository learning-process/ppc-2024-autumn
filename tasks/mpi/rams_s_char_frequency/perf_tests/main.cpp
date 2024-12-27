// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/rams_s_char_frequency/include/ops_mpi.hpp"

TEST(rams_s_char_frequency_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::string common_string = "abc";
  std::string global_in;
  for (int i = 0; i < 999999; i++) {
    global_in += common_string;
  }
  std::vector<int> global_in_target(1, 'a');
  std::vector<int> global_out(1, 0);
  int expected_count = 999999;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in.data()));
    taskDataPar->inputs_count.emplace_back(global_in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in_target.data()));
    taskDataPar->inputs_count.emplace_back(global_in_target.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  auto testMpiTaskParallel = std::make_shared<rams_s_char_frequency_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(expected_count, global_out[0]);
  }
}

TEST(rams_s_char_frequency_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::string common_string = "abc";
  std::string global_in;
  for (int i = 0; i < 999999; i++) {
    global_in += common_string;
  }
  std::vector<int> global_in_target(1, 'a');
  std::vector<int> global_out(1, 0);
  int expected_count = 999999;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in.data()));
    taskDataPar->inputs_count.emplace_back(global_in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_in_target.data()));
    taskDataPar->inputs_count.emplace_back(global_in_target.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  auto testMpiTaskParallel = std::make_shared<rams_s_char_frequency_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(expected_count, global_out[0]);
  }
}
