// Copyright 2024 Koshkin Matvey
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/koshkin_m_dining_philosophers/include/ops_mpi.hpp"

TEST(koshkin_m_dining_philosophers, test_pipeline_run) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  auto testMPI = std::make_shared<koshkin_m_dining_philosophers::testMpiTaskParallel>(taskData);
  if (num_philosophers >= 2) {
    ASSERT_TRUE(testMPI->validation());
    testMPI->pre_processing();
    testMPI->run();
    testMPI->post_processing();

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMPI);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);

    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
      bool deadlock_detected = testMPI->check_deadlock();
      ASSERT_FALSE(deadlock_detected);
    }
  } else
    ASSERT_FALSE(testMPI->validation());
}

TEST(koshkin_m_dining_philosophers, test_task_run) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  auto testMPI = std::make_shared<koshkin_m_dining_philosophers::testMpiTaskParallel>(taskData);
  if (num_philosophers >= 2) {
    ASSERT_TRUE(testMPI->validation());

    testMPI->pre_processing();
    testMPI->run();
    testMPI->post_processing();

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMPI);
    perfAnalyzer->task_run(perfAttr, perfResults);

    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
      bool deadlock_detected = testMPI->check_deadlock();
      ASSERT_FALSE(deadlock_detected);
    }
  } else
    ASSERT_FALSE(testMPI->validation());
}
