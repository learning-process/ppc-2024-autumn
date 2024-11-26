// Copyright 2024 Ivanov Mike
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/ivanov_m_integration_trapezoid/include/ops_mpi.hpp"

TEST(ivanov_m_integration_trapezoid_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  double a = 0;
  double b = 1;
  int n = 1000;

  // Create function y = x^2
  std::function<double(double)> _f = [](double x) { return x * x; };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto testMpiTaskParallel = std::make_shared<ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->add_function(_f);

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
    double result = 1.0 / 3.0;
    ASSERT_NEAR(result, global_result[0], 1e-3);
  }
}

TEST(ivanov_m_integration_trapezoid_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  double a = 0;
  double b = 1;
  int n = 1000;

  // Create function y = x^2
  std::function<double(double)> _f = [](double x) { return x * x; };

  std::vector<double> global_vec = {a, b, static_cast<double>(n)};
  std::vector<double> global_result(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto testMpiTaskParallel = std::make_shared<ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->add_function(_f);

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
    double result = 1.0 / 3.0;
    ASSERT_NEAR(result, global_result[0], 1e-3);
  }
}