// Golovkin Maksim Task#2

#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/golovkin_rowwise_matrix_partitioning/include/ops_mpi.hpp"

using namespace golovkin_rowwise_matrix_partitioning;
using ppc::core::Perf;
using ppc::core::TaskData;

TEST(golovkin_rowwise_matrix_partitioning, test_pipeline_run) {
  boost::mpi::communicator world;
  double *A = nullptr;
  double *B = nullptr;
  double *result = nullptr;
  int rows_A = 700;
  int cols_A = 800;
  int rows_B = 800;
  int cols_B = 300;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    A = new double[rows_A * cols_A]();
    B = new double[rows_B * cols_B]();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B));

    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);

    result = new double[rows_A * cols_B];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result));
    taskDataPar->outputs_count.emplace_back(rows_A);
    taskDataPar->outputs_count.emplace_back(cols_B);
  }

  auto testMpiTaskParallel =
      std::make_shared<golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask>(taskDataPar);
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
  if (world.size() < 5 || world.rank() >= 4) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto *expected_res = new double[rows_A * cols_B]();
    for (int i = 0; i < rows_A * cols_B; i++) {
      ASSERT_NEAR(expected_res[i], result[i], 1e-6);
    }
    delete[] expected_res;
    delete[] result;
    delete[] A;
    delete[] B;
  }
}

TEST(golovkin_rowwise_matrix_partitioning, test_task_run) {
  boost::mpi::communicator world;
  double *A = nullptr;
  double *B = nullptr;
  double *result = nullptr;
  int rows_A = 700;
  int cols_A = 800;
  int rows_B = 800;
  int cols_B = 300;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    A = new double[rows_A * cols_A]();
    B = new double[rows_B * cols_B]();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B));

    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);

    result = new double[rows_A * cols_B];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result));
    taskDataPar->outputs_count.emplace_back(rows_A);
    taskDataPar->outputs_count.emplace_back(cols_B);
  }

  auto testMpiTaskParallel =
      std::make_shared<golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask>(taskDataPar);
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
  if (world.size() < 5 || world.rank() >= 4) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto *expected_res = new double[rows_A * cols_B]();
    for (int i = 0; i < rows_A * cols_B; i++) {
      ASSERT_NEAR(expected_res[i], result[i], 1e-6);
    }
    delete[] expected_res;
    delete[] result;
    delete[] A;
    delete[] B;
  }
}