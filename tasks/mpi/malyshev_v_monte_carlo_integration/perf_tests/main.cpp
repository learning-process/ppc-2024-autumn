#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

TEST(malyshev_v_monte_carlo_integration_mpi, test_monte_carlo_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  double a = -1.0;
  double b = 1.0;
  int num_samples = 100000;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_samples));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  auto MonteCarloIntegrationParallel =
      std::make_shared<malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel>(taskDataPar);
  ASSERT_EQ(MonteCarloIntegrationParallel->validation(), true);
  MonteCarloIntegrationParallel->pre_processing();
  MonteCarloIntegrationParallel->run();
  MonteCarloIntegrationParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MonteCarloIntegrationParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_samples));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));

    malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential MonteCarloIntegrationSequential(taskDataSeq);
    ASSERT_EQ(MonteCarloIntegrationSequential.validation(), true);
    MonteCarloIntegrationSequential.pre_processing();
    MonteCarloIntegrationSequential.run();
    MonteCarloIntegrationSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

TEST(malyshev_v_monte_carlo_integration_mpi, test_monte_carlo_task_run) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  double a = -1.0;
  double b = 1.0;
  int num_samples = 100000;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_samples));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  auto MonteCarloIntegrationParallel =
      std::make_shared<malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel>(taskDataPar);
  ASSERT_EQ(MonteCarloIntegrationParallel->validation(), true);
  MonteCarloIntegrationParallel->pre_processing();
  MonteCarloIntegrationParallel->run();
  MonteCarloIntegrationParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MonteCarloIntegrationParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double reference_result = 2.0;  // Ожидаемое значение интеграла
    ASSERT_NEAR(reference_result, global_result[0], 1e-1);
  }
}
