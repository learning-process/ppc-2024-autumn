// Copyright 2024 Your Name
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/naumov_b_min_colum_matrix/include/ops_mpi.hpp"

static std::vector<std::vector<int>> getRandomMatrix(int rows, int columns) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(columns));
  for (auto& row : matrix) {
    for (int& element : row) {
      element = rand() % 201 - 100;
    }
  }
  return matrix;
}

TEST(naumov_b_min_colum_matrix_mpi_perf, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_results;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  int rows = 1000;
  int cols = 1000;
  if (world.rank() == 0) {
    auto matrix = getRandomMatrix(rows, cols);
    global_matrix.resize(rows * cols);
    global_results.resize(cols);

    for (int i = 0; i < rows; ++i) {
      std::copy(matrix[i].begin(), matrix[i].end(), global_matrix.begin() + i * cols);
    }

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskData->inputs_count = {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_results.data()));
    taskData->outputs_count.push_back(static_cast<uint32_t>(global_results.size()));
  }

  auto task = std::make_shared<naumov_b_min_colum_matrix_mpi::TestMPITaskParallel>(taskData);
  ASSERT_EQ(task->validation(), true);
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(global_results.size(), static_cast<size_t>(cols));
  }
}

TEST(naumov_b_min_colum_matrix_mpi, test_column_minimum_task_run) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  const int rows = 100;
  const int cols = 100;

  std::vector<std::vector<int>> matrix = getRandomMatrix(rows, cols);

  if (world.rank() == 0) {
    std::vector<int> flatMatrix(rows * cols);
    for (int i = 0; i < rows; ++i) {
      std::copy(matrix[i].begin(), matrix[i].end(), flatMatrix.begin() + i * cols);
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(flatMatrix.data()));
    taskDataPar->inputs_count = {rows, cols};

    std::vector<int> output_results(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_results.data()));
    taskDataPar->outputs_count = {static_cast<uint32_t>(output_results.size())};
  }

  auto testMpiTaskParallel = std::make_shared<naumov_b_min_colum_matrix_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
}
