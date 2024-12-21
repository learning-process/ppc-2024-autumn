#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/tyshkevich_a_monte_carlo/include/ops_mpi.hpp"
#include "mpi/tyshkevich_a_monte_carlo/include/test_include.hpp"

TEST(tyshkevich_a_monte_carlo_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::function<double(const std::vector<double>&)> function = tyshkevich_a_monte_carlo_mpi::function_sin_sum;
  double exp_res = 1.379093;
  int dimensions = 3;
  double precision = 1000000;
  double left_bound = 0.0;
  double right_bound = 1.0;
  double result = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dimensions));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&precision));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left_bound));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right_bound));

  if (world.rank() == 0) {
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  }

  auto task = std::make_shared<tyshkevich_a_monte_carlo_mpi::MonteCarloParallelMPI>(taskDataPar, std::move(function));

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_NEAR(result, exp_res, 0.1);
  }
}

TEST(tyshkevich_a_monte_carlo_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::function<double(const std::vector<double>&)> function = tyshkevich_a_monte_carlo_mpi::function_sin_sum;
  double exp_res = 1.379093;
  int dimensions = 3;
  double precision = 1000000;
  double left_bound = 0.0;
  double right_bound = 1.0;
  double result = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dimensions));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&precision));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left_bound));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right_bound));

  if (world.rank() == 0) taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  auto task = std::make_shared<tyshkevich_a_monte_carlo_mpi::MonteCarloParallelMPI>(taskDataPar, std::move(function));

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_NEAR(result, exp_res, 0.1);
  }
}
