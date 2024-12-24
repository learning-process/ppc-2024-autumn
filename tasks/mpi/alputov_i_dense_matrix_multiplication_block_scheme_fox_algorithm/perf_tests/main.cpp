#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <boost/serialization/map.hpp>
#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm/include/ops_mpi.hpp"

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int x = static_cast<int>(std::sqrt(static_cast<double>(world.size())));
  if (x * x != world.size()) {
    GTEST_SKIP();
  }

  int N = 100 * x;
  std::vector<double> A(N * N, 0);
  std::vector<double> B(N * N, 0);
  std::vector<double> out(N * N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
                           dense_matrix_multiplication_block_scheme_fox_algorithm_mpi>(taskDataPar);

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
    for (int i = 0; i < N * N; ++i) {
      ASSERT_NEAR(0, out[0], 1e-5);
    }
  }
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_mpi, test_task_run) {
  boost::mpi::communicator world;
  int x = static_cast<int>(std::sqrt(static_cast<double>(world.size())));
  if (x * x != world.size()) {
    GTEST_SKIP();
  }

  int N = 100 * x;
  std::vector<double> A(N * N, 0);
  std::vector<double> B(N * N, 0);
  std::vector<double> out(N * N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
                           dense_matrix_multiplication_block_scheme_fox_algorithm_mpi>(taskDataPar);
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
    for (int i = 0; i < N * N; ++i) {
      ASSERT_NEAR(0, out[0], 1e-5);
    }
  }
}
