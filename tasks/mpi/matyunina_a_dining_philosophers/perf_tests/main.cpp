// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/matyunina_a_dining_philosophers/include/ops_mpi.hpp"

TEST(matyunina_a_dining_philosophers_mpi, test_1) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(1, 1);
  std::vector<int32_t> average_value(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(average_value.data()));
    taskDataPar->outputs_count.emplace_back(average_value.size());
  }

  auto testMpiTaskParallel = std::make_shared<matyunina_a_dining_philosophers_mpi::TestMPITaskParallel>(taskDataPar);
  if (world.size() < 3) {
    if (world.rank() == 0) {
      ASSERT_EQ(testMpiTaskParallel->validation(), false);
    }
  } else {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();
    if (world.rank() == 0) {
      ASSERT_EQ(global_vec[0] * (world.size() - 1), average_value[0]);
    }

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
      ASSERT_EQ(global_vec[0] * (world.size() - 1), average_value[0]);
    }
  }
}
TEST(matyunina_a_dining_philosophers_mpi, test_2) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(1, 1);
  std::vector<int32_t> average_value(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(average_value.data()));
    taskDataPar->outputs_count.emplace_back(average_value.size());
  }

  auto testMpiTaskParallel = std::make_shared<matyunina_a_dining_philosophers_mpi::TestMPITaskParallel>(taskDataPar);
  if (world.size() < 3) {
    if (world.rank() == 0) {
      ASSERT_EQ(testMpiTaskParallel->validation(), false);
    }
  } else {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();
    if (world.rank() == 0) {
      ASSERT_EQ(global_vec[0] * (world.size() - 1), average_value[0]);
    }

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
      ASSERT_EQ(global_vec[0] * (world.size() - 1), average_value[0]);
    }
  }
}