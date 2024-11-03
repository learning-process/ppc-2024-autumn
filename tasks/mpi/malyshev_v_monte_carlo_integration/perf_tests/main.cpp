#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

TEST(malyshev_v_monte_carlo_integration_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  double a = 0.0, b = 1.0;
  int n = 1000000;
  double global_result = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto mpiTaskParallel =
      std::make_shared<malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel>(taskDataPar);
  ASSERT_TRUE(mpiTaskParallel->validation());
  mpiTaskParallel->pre_processing();
  mpiTaskParallel->run();
  mpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_res = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyz = std::make_shared<ppc::core::Perf>(mpiTaskParallel);
  perf_analyz->pipeline_run(perfAttr, perf_res);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_res);
    ASSERT_NEAR(global_result, 0.5, 0.01);  // Expected result for integration of f(x) = x over [0, 1]
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, test_task_run) {
  boost::mpi::communicator world;
  double a = 0.0, b = 1.0;
  int n = 1000000;
  double global_result = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_result));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto mpiTaskParallel =
      std::make_shared<malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel>(taskDataPar);
  ASSERT_TRUE(mpiTaskParallel->validation());
  mpiTaskParallel->pre_processing();
  mpiTaskParallel->run();
  mpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_res = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyz = std::make_shared<ppc::core::Perf>(mpiTaskParallel);
  perf_analyz->task_run(perfAttr, perf_res);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_res);
    ASSERT_NEAR(global_result, 0.5, 0.01);  // Expected result for integration of f(x) = x over [0, 1]
  }
}
