#include <boost/mpi/timer.hpp>

#include "../include/fox_algorithm.hpp"
#include "../include/tests.hpp"
#include "core/perf/include/perf.hpp"

using namespace khasanyanov_k_fox_algorithm;

TEST(khasanyanov_k_fox_algorithm_tests, test_pipeline_run) {
  boost::mpi::communicator world;
  const int m = 256;
  const int n = 256;

  matrix<double> A = MatrixOperations::generate_random_matrix<double>(m, n, -1000, 1000);
  matrix<double> B = MatrixOperations::generate_random_matrix<double>(m, n, -1000, 1000);
  matrix<double> C{m, n};
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = create_task_data(A, B, C);
  }
  auto test = std::make_shared<FoxAlgorithm<double>>(taskData);
  RUN_TASK(*test);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(khasanyanov_k_fox_algorithm_tests, test_task_run) {
  boost::mpi::communicator world;
  const int m = 256;
  const int n = 256;

  matrix<double> A = MatrixOperations::generate_random_matrix<double>(m, n, -1000, 1000);
  matrix<double> B = MatrixOperations::generate_random_matrix<double>(m, n, -1000, 1000);
  matrix<double> C{m, n};
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = create_task_data(A, B, C);
  }
  auto test = std::make_shared<FoxAlgorithm<double>>(taskData);
  RUN_TASK(*test);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}