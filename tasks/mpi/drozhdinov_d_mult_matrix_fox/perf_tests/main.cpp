// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
// not example
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/drozhdinov_d_mult_matrix_fox/include/ops_mpi.hpp"
using namespace drozhdinov_d_mult_matrix_fox_mpi;
namespace drozhdinov_d_mult_matrix_fox_mpi {
std::vector<double> MatrixMult(const std::vector<double> &A, const std::vector<double> &B, int k, int l, int n) {
  std::vector<double> result(k * n, 0.0);

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      for (int p = 0; p < l; p++) {
        result[i * n + j] += A[i * l + p] * B[p * n + j];
      }
    }
  }

  return result;
}

std::vector<double> getRandomMatrix(int sz, int lbound, int rbound) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> vec(sz);
  std::uniform_int_distribution<int> dist(lbound, rbound);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace drozhdinov_d_mult_matrix_fox_mpi

TEST(drozhdinov_d_mult_matrix_fox_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  int k = 250;
  int l = 250;
  int m = 250;
  int n = 250;
  std::vector<double> A = getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = getRandomMatrix(m * n, -100, 100);
  std::vector<double> expres_par(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(l);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(k);
    taskDataPar->outputs_count.emplace_back(n);
  }

  auto testMpiTaskParallel = std::make_shared<drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 50;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < k * n; i++) {
      EXPECT_DOUBLE_EQ(expres_par[i], expres[i]);
      EXPECT_DOUBLE_EQ(expres_par[i], expres[i]);
    }
  }
}

TEST(drozhdinov_d_mult_matrix_fox_perf_test, test_task_run) {
  boost::mpi::communicator world;
  int k = 250;
  int l = 250;
  int m = 250;
  int n = 250;
  std::vector<double> A = getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = getRandomMatrix(m * n, -100, 100);
  std::vector<double> expres_par(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(k);
    taskDataPar->inputs_count.emplace_back(l);
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(k);
    taskDataPar->outputs_count.emplace_back(n);
  }

  auto testMpiTaskParallel = std::make_shared<drozhdinov_d_mult_matrix_fox_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 50;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < k * n; i++) {
      EXPECT_DOUBLE_EQ(expres_par[i], expres[i]);
      EXPECT_DOUBLE_EQ(expres_par[i], expres[i]);
    }
  }
}