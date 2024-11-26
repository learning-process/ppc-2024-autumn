#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kozlova_e_jacobi_method/include/ops_mpi.hpp"

TEST(kozlova_e_jacobi_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int N = 890;
  std::vector<double> A(N * N, 5.1231234);
  for (int i = 0; i < N; ++i) {
    A[i * N + i] = 5000.0;
  }
  std::vector<double> B(N, 5000.233445);
  std::vector<double> X(N, 0.0);
  std::vector<double> expected_X(N, 1.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());
  }

  auto testMpiTaskParallel = std::make_shared<kozlova_e_jacobi_method_mpi::MethodJacobiMPI>(taskDataPar);
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
    for (int i = 0; i < N; ++i) {
      ASSERT_NEAR(X[i], expected_X[i], 0.5);
    }
  }
}

TEST(kozlova_e_jacobi_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int N = 490;
  std::vector<double> A(N * N, 1.0);
  for (int i = 0; i < N; ++i) {
    A[i * N + i] = 510.0;
  }
  std::vector<double> B(N, 510.0);
  std::vector<double> X(N, 0.0);
  std::vector<double> expected_X(N, 1.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());
  }

  auto testMpiTaskParallel = std::make_shared<kozlova_e_jacobi_method_mpi::MethodJacobiMPI>(taskDataPar);
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
  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(X[i], expected_X[i], 0.5);
  }
}
