// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/rysev_m_count_of_sent/include/ops_mpi.hpp"

TEST(rysev_m_count_of_sent_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  //create data
  std::string str = "Have I told you what madness is?";
  std::vector<int> out(1, 0);
  for (int i = 0; i < 100; i++) str.append("Have I told you what madness is?");

  //create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataPar->inputs_count.emplace_back(str.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataPar->outputs_count.emplace_back(out.size());

  auto testParTask = std::make_shared<rysev_m_count_of_sent_mpi::CountOfSentParallel>(taskDataPar);
  ASSERT_EQ(testParTask->validation(), true);
  testParTask->pre_processing();
  testParTask->run();
  testParTask->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testParTask);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(101, out[0]);
  }
}

TEST(rysev_m_count_of_sent_mpi, test_task_run) {
  boost::mpi::communicator world;

  // create data
  std::string str = "Have I told you what madness is?";
  std::vector<int> out(1, 0);
  for (int i = 0; i < 100; i++) str.append("Have I told you what madness is?");

  // create taskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str.data()));
  taskDataPar->inputs_count.emplace_back(str.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataPar->outputs_count.emplace_back(out.size());

  auto testParTask = std::make_shared<rysev_m_count_of_sent_mpi::CountOfSentParallel>(taskDataPar);
  ASSERT_EQ(testParTask->validation(), true);
  testParTask->pre_processing();
  testParTask->run();
  testParTask->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testParTask);

  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(101, out[0]);
  }
}