#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shuravina_o_monte_carlo/include/ops_mpi.hpp"

TEST(MonteCarloIntegrationTaskParallel, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> out(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel>(taskDataPar);
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
  }
}

TEST(MonteCarloIntegrationTaskParallel, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> out(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(nullptr);
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel>(taskDataPar);
  testMpiTaskParallel->set_interval(0.0, 1.0);
  testMpiTaskParallel->set_num_points(100000000);
  testMpiTaskParallel->set_function([](double x) { return x * x; });

  std::cout << "Rank " << world.rank() << " is validating." << std::endl;
  ASSERT_EQ(testMpiTaskParallel->validation(), true);

  std::cout << "Rank " << world.rank() << " is pre-processing." << std::endl;
  testMpiTaskParallel->pre_processing();

  std::cout << "Rank " << world.rank() << " is running." << std::endl;
  testMpiTaskParallel->run();

  std::cout << "Rank " << world.rank() << " is post-processing." << std::endl;
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);

  std::cout << "Rank " << world.rank() << " is processing pipeline." << std::endl;
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    std::cout << "Rank " << world.rank() << " is printing performance statistics." << std::endl;
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}