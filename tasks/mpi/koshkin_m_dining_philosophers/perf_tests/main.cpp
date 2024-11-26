// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/koshkin_m_dining_philosophers/include/ops_mpi.hpp"

TEST(koshkin_m_dining_philosophers, test_pipeline_run) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int num_philosophers = world.size();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_philosophers));
    taskDataPar->inputs_count.emplace_back(sizeof(int));
  }

  auto testMpiTaskParallel = std::make_shared<koshkin_m_dining_philosophers::TestMPITaskParallel>(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel->validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel->validation());
  }

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    int expected_philosophers_finished = world.size();
    ASSERT_EQ(expected_philosophers_finished, world.size());
  }
}

TEST(koshkin_m_dining_philosophers, test_task_run) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int num_philosophers = world.size();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_philosophers));
    taskDataPar->inputs_count.emplace_back(sizeof(int));
  }

  auto testMpiTaskParallel = std::make_shared<koshkin_m_dining_philosophers::TestMPITaskParallel>(taskDataPar);

  if (world.size() < 2) {
    ASSERT_FALSE(testMpiTaskParallel->validation());
  } else {
    ASSERT_TRUE(testMpiTaskParallel->validation());
  }

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    int expected_philosophers_finished = world.size();
    ASSERT_EQ(expected_philosophers_finished, world.size());
  }
}
