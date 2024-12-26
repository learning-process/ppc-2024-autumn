#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <numbers>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vedernikova_k_monte_carlo_integration_mpi/include/ops_mpi.hpp"

TEST(vedernikova_k_monte_carlo_integration_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  auto [ax, bx] = std::make_pair(0.0, 2.0);
  auto [ay, by] = std::make_pair(0.0, std::numbers::pi);
  auto [az, bz] = std::make_pair(0.0, std::numbers::pi);
  size_t num_point = 700000;

  double out = 0.0;
  double expected_res = 128 * std::numbers::pi / 15;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&az));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bz));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  }

  auto testMpiTaskParallel =
      std::make_shared<vedernikova_k_monte_carlo_integration_mpi::TestMPITaskParallel>(taskDataPar);

  testMpiTaskParallel->f = [](double x, double y, double z) { return x * x + y * y; };
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
    EXPECT_NEAR(expected_res, out, 2.0);
  }
}

TEST(vedernikova_k_monte_carlo_integration_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  auto [ax, bx] = std::make_pair(0.0, 2.0);
  auto [ay, by] = std::make_pair(0.0, std::numbers::pi);
  auto [az, bz] = std::make_pair(0.0, std::numbers::pi);
  size_t num_point = 700000;

  double out = 0.0;
  double expected_res = 128 * std::numbers::pi / 15;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&az));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bz));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  }

  auto testMpiTaskParallel =
      std::make_shared<vedernikova_k_monte_carlo_integration_mpi::TestMPITaskParallel>(taskDataPar);

  testMpiTaskParallel->f = [](double x, double y, double z) { return x * x + y * y; };
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
    EXPECT_NEAR(expected_res, out, 2.0);
  }
}
