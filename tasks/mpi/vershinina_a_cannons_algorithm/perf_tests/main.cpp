
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vershinina_a_cannons_algorithm/include/ops_mpi.hpp"

std::vector<double> getRandomMatrix(double r) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(0, 100);
  std::vector<double> matrix(r * r, 0.0);
  for (int i = 0; i < r * r; i++) {
    matrix[i] = distr(gen);
  }
  return matrix;
}

TEST(vershinina_a_cannons_algorithm, test_pipeline_run) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }
  int n = 100;
  auto lhs = getRandomMatrix(100);
  auto rhs = getRandomMatrix(100);

  std::vector<double> res(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lhs.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  }

  auto testMpiTaskParallel = std::make_shared<vershinina_a_cannons_algorithm::TestMPITaskParallel>(taskDataPar);
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

TEST(vershinina_a_cannons_algorithm, test_task_run) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }
  int n = 100;
  auto lhs = getRandomMatrix(100);
  auto rhs = getRandomMatrix(100);

  std::vector<double> res(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lhs.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  }

  auto testMpiTaskParallel = std::make_shared<vershinina_a_cannons_algorithm::TestMPITaskParallel>(taskDataPar);
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
  }
}