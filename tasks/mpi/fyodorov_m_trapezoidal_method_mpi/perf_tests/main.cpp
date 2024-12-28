// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/fyodorov_m_trapezoidal_method_mpi/include/ops_mpi.hpp"

TEST(fyodorov_m_trapezoidal_method_mpi_perf_test, test_pipeline_runs) {
  boost::mpi::communicator world;

  std::function<double(const std::vector<double>&)> func = [](const std::vector<double>& point) { return 1.0; };

  std::vector<double> lower_bounds = {0.0};
  std::vector<double> upper_bounds = {1.0};
  std::vector<int> intervals = {1000000};

  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataPar->inputs_count = {1, 1, 1};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count = {1};
  }

  auto testMpiTaskParallel =
      std::make_shared<fyodorov_m_trapezoidal_method_mpi::TestMPITaskParallel>(taskDataPar, func);
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

    double expected_result = 1.0;
    ASSERT_NEAR(global_result[0], expected_result, 1e-6);
  }
}

TEST(fyodorov_m_trapezoidal_method_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;

  std::function<double(const std::vector<double>&)> func = [](const std::vector<double>& point) { return 1.0; };

  std::vector<double> lower_bounds = {0.0};
  std::vector<double> upper_bounds = {1.0};
  std::vector<int> intervals = {1000000};

  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataPar->inputs_count = {1, 1, 1};

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count = {1};
  }

  auto testMpiTaskParallel =
      std::make_shared<fyodorov_m_trapezoidal_method_mpi::TestMPITaskParallel>(taskDataPar, func);
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

    double expected_result = 1.0;
    ASSERT_NEAR(global_result[0], expected_result, 1e-6);
  }
}
