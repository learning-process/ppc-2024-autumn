// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/grudzin_k_monte_carlo/include/gmc_include.hpp"

TEST(grudzin_k_monte_carlo_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int dimensions = 3;
  int N = 5000000;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    { return 0.0 * (x[0] + x[1] + x[2]) + 1.0; }
  };
  std::shared_ptr<ppc::core::TaskData> MC1_Data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> dim = {0, 1, 0, 1, 0, 1};
  double result_par = 0;
  double result_seq = 1;

  if (world.rank() == 0) {
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC1_Data->inputs_count.emplace_back(dimensions);
    MC1_Data->inputs_count.emplace_back(1);
    MC1_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_par));
    MC1_Data->outputs_count.emplace_back(1);
  }
  auto testMpiTaskMyRealization = std::make_shared<grudzin_k_montecarlo_mpi::MonteCarloMpi<dimensions>>(MC1_Data, f);
  ASSERT_EQ(testMpiTaskMyRealization->validation(), true);
  testMpiTaskMyRealization->pre_processing();
  testMpiTaskMyRealization->run();
  testMpiTaskMyRealization->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 20;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskMyRealization);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_LE(abs(result_par - result_seq) / result_seq, 1e-1);
  }
}

TEST(grudzin_k_monte_carlo_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int dimensions = 3;
  int N = 5000000;
  std::function<double(std::array<double, dimensions> &)> f = [](std::array<double, dimensions> &x) {
    { return 0.0 * (x[0] + x[1] + x[2]) + 1.0; }
  };
  std::shared_ptr<ppc::core::TaskData> MC1_Data = std::make_shared<ppc::core::TaskData>();
  std::vector<double> dim = {0, 1, 0, 1, 0, 1};
  double result_par = 0;
  double result_seq = 1;

  if (world.rank() == 0) {
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(dim.data()));
    MC1_Data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
    MC1_Data->inputs_count.emplace_back(dimensions);
    MC1_Data->inputs_count.emplace_back(1);
    MC1_Data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&result_par));
    MC1_Data->outputs_count.emplace_back(1);
  }
  auto testMpiTaskMyRealization = std::make_shared<grudzin_k_montecarlo_mpi::MonteCarloMpi<dimensions>>(MC1_Data, f);
  ASSERT_EQ(testMpiTaskMyRealization->validation(), true);
  testMpiTaskMyRealization->pre_processing();
  testMpiTaskMyRealization->run();
  testMpiTaskMyRealization->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 20;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskMyRealization);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_LE(abs(result_par - result_seq) / result_seq, 1e-1);
  }
}