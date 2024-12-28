// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/varfolomeev_g_matrix_max_rows_vals/include/ops_mpi.hpp"

namespace varfolomeev_g_matrix_max_rows_vals_mpi {
static void fillVectorBetween(std::vector<int> &vec, int a, int b) {
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (size_t i = 0; i < vec.size(); i++) {
    vec[i] = std::rand() % (b - a + 1) + a;
  }
}
static int searchMaxInVec(const std::vector<int> &matrix, int row, int cols) {
  int max = matrix[row * cols];
  for (int j = 1; j < cols; j++) {
    if (max < matrix[row * cols + j]) max = matrix[row * cols + j];
  }
  return max;
}
}  // namespace varfolomeev_g_matrix_max_rows_vals_mpi

TEST(mpi_varfolomeev_g_matrix_max_rows_perf_test, test_pipeline_run) {
  int size_m = 3000;
  int size_n = 3000;
  int a = -100;
  int b = 100;
  boost::mpi::communicator world;

  std::vector<int> matrix(size_m * size_n);
  varfolomeev_g_matrix_max_rows_vals_mpi::fillVectorBetween(matrix, a, b);
  std::vector<int32_t> max_vec(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Setting rows and cols

  // If curr. proc. is root (r.0), setting the input and output data
  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(max_vec.data()));
    taskDataPar->outputs_count.emplace_back(max_vec.size());
  }
  auto testMpiTaskParallel = std::make_shared<varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel>(taskDataPar);
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
  // If curr. proc. is root (r.0), display performance and check the result
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  if (world.rank() == 0) {
    for (int i = 0; i < size_m; ++i) {
      int expected_max = varfolomeev_g_matrix_max_rows_vals_mpi::searchMaxInVec(matrix, i, size_n);
      EXPECT_EQ(max_vec[i], expected_max);
    }
  }
}

TEST(mpi_varfolomeev_g_matrix_max_rows_perf_test, test_task_run) {
  int size_m = 3000;
  int size_n = 3000;
  int a = -100;
  int b = 100;
  boost::mpi::communicator world;

  std::vector<int> matrix(size_m * size_n);
  varfolomeev_g_matrix_max_rows_vals_mpi::fillVectorBetween(matrix, a, b);
  std::vector<int32_t> max_vec(size_m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(size_m);
  taskDataPar->inputs_count.emplace_back(size_n);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(max_vec.data()));
    taskDataPar->outputs_count.emplace_back(max_vec.size());
  }
  auto testMpiTaskParallel = std::make_shared<varfolomeev_g_matrix_max_rows_vals_mpi::MaxInRowsParallel>(taskDataPar);
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
  ppc::core::Perf::print_perf_statistic(perfResults);
  if (world.rank() == 0) {
    for (int i = 0; i < size_m; ++i) {
      int expected_max = varfolomeev_g_matrix_max_rows_vals_mpi::searchMaxInVec(matrix, i, size_n);
      EXPECT_EQ(max_vec[i], expected_max);
    }
  }
}