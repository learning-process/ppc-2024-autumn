#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/kazunin_n_dining_philosophers/include/ops_mpi.hpp"
#include "mpi/kazunin_n_dining_philosophers/src/ops_mpi.cpp"

TEST(KazuninDiningPhilosophersPerfTest, TestPipelineRun) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  auto diningTask = std::make_shared<kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<int>>(taskData);
  if (num_philosophers >= 2) {
    ASSERT_TRUE(diningTask->validation());
    diningTask->pre_processing();
    diningTask->run();
    diningTask->post_processing();

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(diningTask);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);

    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
    }
  } else
    ASSERT_FALSE(diningTask->validation());
}

TEST(KazuninDiningPhilosophersPerfTest, TestTaskRun) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  auto diningTask = std::make_shared<kazunin_n_dining_philosophers::KazuninDiningPhilosophersMPI<int>>(taskData);
  if (num_philosophers >= 2) {
    ASSERT_TRUE(diningTask->validation());

    diningTask->pre_processing();
    diningTask->run();
    diningTask->post_processing();

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(diningTask);
    perfAnalyzer->task_run(perfAttr, perfResults);

    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
    }
  } else
    ASSERT_FALSE(diningTask->validation());
}
