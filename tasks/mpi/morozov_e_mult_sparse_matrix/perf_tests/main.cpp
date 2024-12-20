// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/morozov_e_mult_sparse_matrix/include/ops_mpi.hpp"

TEST(morozov_e_mult_sparse_matrix_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrixA = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<std::vector<double>> matrixB = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  if (world.rank() == 0) {
    morozov_e_mult_sparse_matrix::fillData(taskData, matrixA.size(), matrixA[0].size(), matrixB.size(),
                                           matrixB[0].size(), dA, row_indA, col_indA, dB, row_indB, col_indB, out);
  }

  auto testMpiTaskParallel = std::make_shared<morozov_e_mult_sparse_matrix::TestMPITaskParallel>(taskData);
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
    std::vector<std::vector<double>> ans(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    for (size_t i = 0; i < out.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskData->outputs[i]);
      ans[i] = std::vector(ptr, ptr + matrixB.size());
    }
    std::vector<std::vector<double>> check_result = {{2, 0, 6}, {0, 14, 0}, {4, 0, 12}};
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(check_result, ans);
  }
}

TEST(morozov_e_mult_sparse_matrix, test_task_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrixA = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<std::vector<double>> matrixB = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  if (world.rank() == 0) {
    morozov_e_mult_sparse_matrix::fillData(taskData, matrixA.size(), matrixA[0].size(), matrixB.size(),
                                           matrixB[0].size(), dA, row_indA, col_indA, dB, row_indB, col_indB, out);
  }

  auto testMpiTaskParallel = std::make_shared<morozov_e_mult_sparse_matrix::TestMPITaskParallel>(taskData);
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
  if (world.rank() == 0) {
    std::vector<std::vector<double>> ans(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    for (size_t i = 0; i < out.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskData->outputs[i]);
      ans[i] = std::vector(ptr, ptr + matrixB.size());
    }
    std::vector<std::vector<double>> check_result = {{2, 0, 6}, {0, 14, 0}, {4, 0, 12}};
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(check_result, ans);
  }
}
