#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/pikarychev_i_monte_carlo/include/ops_mpi.hpp"

TEST(mpi_pikarychev_i_monte_carlo_perf_test, test_monte_carlo_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> global_result;
  std::vector<double> expected_result(1, 0.33335967263763133);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    double a = 0.0;
    double b = 1.0;
    int num_samples = 10000000;
    int seed = 12345;
    global_result = {0.0};
    std::vector<double> inputs = {a, b, static_cast<double>(num_samples), static_cast<double>(seed)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
    taskDataPar->inputs_count.emplace_back(inputs.size());
    taskDataPar->outputs_count.emplace_back(global_result.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  auto testMpiTaskParallel = std::make_shared<pikarychev_i_monte_carlo_parallel::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_NEAR(expected_result[0], global_result[0], 0.1);
  }
}

TEST(mpi_pikarychev_i_monte_carlo_perf_test, test_monte_carlo_task_run) {
  boost::mpi::communicator world;
  std::vector<double> global_result;
  std::vector<double> expected_result(1, 0.33335967263763133);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    double a = 0.0;
    double b = 1.0;
    int num_samples = 1000000;
    int seed = 12345;
    global_result = {0.0};
    std::vector<double> inputs = {a, b, static_cast<double>(num_samples), static_cast<double>(seed)};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs.data()));
    taskDataPar->inputs_count.emplace_back(inputs.size());
    taskDataPar->outputs_count.emplace_back(global_result.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  auto testMpiTaskParallel = std::make_shared<pikarychev_i_monte_carlo_parallel::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_NEAR(expected_result[0], global_result[0], 0.1);
  }
}