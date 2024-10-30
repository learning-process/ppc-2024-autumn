#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/nikolaev_r_trapezoidal_integral/include/ops_mpi.hpp"

TEST(nikolaev_r_trapezoidal_integral_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  double a = -2.0;
  double b = 10.0;
  int n = 100000;
  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralParallel>(taskDataPar);
  auto f = [](double x) { return std::pow(x, 3) - std::pow(3, x) + std::exp(x); };
  testMpiTaskParallel->set_function(f);

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
    double accurate_result = -29226.28;
    ASSERT_NEAR(accurate_result, result, 0.01);
  }
}

TEST(nikolaev_r_trapezoidal_integral_mpi, test_task_run) {
  boost::mpi::communicator world;
  double a = -2.0;
  double b = 10.0;
  int n = 100000;
  double result = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<nikolaev_r_trapezoidal_integral_mpi::TrapezoidalIntegralParallel>(taskDataPar);
  auto f = [](double x) { return std::pow(x, 3) - std::pow(3, x) + std::exp(x); };
  testMpiTaskParallel->set_function(f);

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
    double accurate_result = -29226.28;
    ASSERT_NEAR(accurate_result, result, 0.01);
  }
}
