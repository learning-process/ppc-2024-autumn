#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/beskhmelnova_k_dinning_philosophers/include/dinning_philosophers.hpp"
#include "mpi/beskhmelnova_k_dinning_philosophers/src/dinning_philosophers.cpp"

TEST(dining_philosophers_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  auto diningTask = std::make_shared<beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int>>(taskData);
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
      bool deadlock_detected = diningTask->check_deadlock();
      ASSERT_FALSE(deadlock_detected);
    }
  } else
    ASSERT_FALSE(diningTask->validation());
}

TEST(dining_philosophers_perf_test, test_task_run) {
  boost::mpi::communicator world;

  int num_philosophers = world.size();
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(num_philosophers);

  auto diningTask = std::make_shared<beskhmelnova_k_dining_philosophers::DiningPhilosophersMPI<int>>(taskData);
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
      bool deadlock_detected = diningTask->check_deadlock();
      ASSERT_FALSE(deadlock_detected);
    }
  } else
    ASSERT_FALSE(diningTask->validation());
}
