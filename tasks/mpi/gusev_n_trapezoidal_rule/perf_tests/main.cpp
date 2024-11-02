#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gusev_n_trapezoidal_rule/include/ops_mpi.hpp"

TEST(gusev_n_trapezoidal_rule_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  double a = 0.0;
  double b = 1.0;
  int n = 100000000;
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.push_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return x * x; });

  ASSERT_TRUE(testMpiTaskParallel->validation());
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
    double exact = 1.0 / 3.0;
    EXPECT_NEAR(output, exact, 1e-4);
  }
}

TEST(gusev_n_trapezoidal_rule_mpi, test_task_run) {
  boost::mpi::communicator world;
  double a = 0.0;
  double b = 1.0;
  int n = 100000000;
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.push_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<gusev_n_trapezoidal_rule_mpi::TrapezoidalIntegrationParallel>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return x * x; });

  ASSERT_TRUE(testMpiTaskParallel->validation());
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
    double exact = 1.0 / 3.0;
    EXPECT_NEAR(output, exact, 1e-4);
  }
}
