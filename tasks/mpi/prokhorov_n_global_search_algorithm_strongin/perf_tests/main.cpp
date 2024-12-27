// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> global_a;
  std::vector<double> global_b;
  std::vector<double> global_epsilon;
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_a = {-10.0};
    global_b = {10.0};
    global_epsilon = {0.001};

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
    taskDataPar->inputs_count.emplace_back(global_a.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
    taskDataPar->inputs_count.emplace_back(global_b.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
    taskDataPar->inputs_count.emplace_back(global_epsilon.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
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
    EXPECT_NEAR(global_result[0], 0.0, 0.001);
  }
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> global_a;
  std::vector<double> global_b;
  std::vector<double> global_epsilon;
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_a = {-10.0};
    global_b = {10.0};
    global_epsilon = {0.001};

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
    taskDataPar->inputs_count.emplace_back(global_a.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
    taskDataPar->inputs_count.emplace_back(global_b.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
    taskDataPar->inputs_count.emplace_back(global_epsilon.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
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
    EXPECT_NEAR(global_result[0], 0.0, 0.001);
  }
}