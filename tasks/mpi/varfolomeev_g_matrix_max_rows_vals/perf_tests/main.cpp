// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/varfolomeev_g_matrix_max_rows_vals/include/ops_mpi.hpp"

namespace varfolomeev_g_matrix_max_rows_vals_mpi {
static std::vector<std::vector<int>> generateMatrix(int rows, int cols, int a, int b) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
  // set generator
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i][j] = std::rand() % (b - a + 1) + a;
    }
  }
  return matrix;
}

static int searchMaxInVec(std::vector<int> vec) {
  int max = vec[0];
  for (size_t i = 1; i < vec.size(); i++) {
    if (max < vec[i]) max = vec[i];
  }
  return max;
}
}  // namespace varfolomeev_g_matrix_max_rows_vals_mpi

TEST(mpi_varfolomeev_g_matrix_max_rows_perf_test, test_pipeline_run) {
  int rows = 6000;
  int cols = 6000;
  int a = -100;
  int b = 100;

  boost::mpi::communicator world;

  std::vector<std::vector<int>> matrix = varfolomeev_g_matrix_max_rows_vals_mpi::generateMatrix(rows, cols, a, b);

  std::vector<int> max_vec(rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Setting rows and cols

  // If curr. proc. is root (r.0), setting the input and output data
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
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

  // Check results
  for (int i = 0; i < rows; ++i) {
    int expected_max = varfolomeev_g_matrix_max_rows_vals_mpi::searchMaxInVec(matrix[i]);
    EXPECT_EQ(max_vec[i], expected_max);
  }
}

TEST(mpi_varfolomeev_g_matrix_max_rows_perf_test, test_task_run) {
  int rows = 6000;
  int cols = 6000;
  int a = -100;
  int b = 100;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> matrix = varfolomeev_g_matrix_max_rows_vals_mpi::generateMatrix(rows, cols, a, b);
  std::vector<int> max_vec(rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(cols);
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
  // Check results
  for (int i = 0; i < rows; ++i) {
    int expected_max = varfolomeev_g_matrix_max_rows_vals_mpi::searchMaxInVec(matrix[i]);
    EXPECT_EQ(max_vec[i], expected_max);
  }
}