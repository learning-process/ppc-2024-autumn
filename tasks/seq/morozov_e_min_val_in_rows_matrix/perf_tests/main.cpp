// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/morozov_e_min_val_in_rows_matrix/include/ops_seq.hpp"
std::vector<std::vector<int>> getRandomMatrix(int n, int m) {
  int left = 0;
  int right = 10005;

  // Создаем матрицу
  std::vector<std::vector<int>> matrix(n, std::vector<int>(m));

  // Заполняем матрицу случайными значениями
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      matrix[i][j] = left + std::rand() % (right - left + 1);
    }
  }
  for (int i = 0; i < n; ++i) {
    int m_ = std::rand() % m;
    matrix[i][m_] = -1;
  }
  return matrix;
}
TEST(sequential_example_perf_test, test_pipeline_run_my) {
  const int n = 5000;
  const int m = 5000;
  std::vector<std::vector<int>> matrix(n, std::vector<int>(m));
  std::vector<int> res(n);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  matrix = getRandomMatrix(n, m);
  for (int i = 0; i < n; ++i) taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
  taskDataPar->inputs_count.emplace_back(n);
  taskDataPar->inputs_count.emplace_back(m);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataPar->outputs_count.emplace_back(n);
  auto testTaskSeq = std::make_shared<morozov_e_min_val_in_rows_matrix::TestTaskSequential>(taskDataPar);
  ASSERT_EQ(testTaskSeq->validation(), true);
  testTaskSeq->pre_processing();
  testTaskSeq->run();
  testTaskSeq->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(-1, res[i]);
  }
}

TEST(sequential_example_perf_test, test_task_run_my) {
  const int n = 5000;
  const int m = 5000;
  std::vector<std::vector<int>> matrix(n, std::vector<int>(m));
  std::vector<int> res(n);
  matrix = getRandomMatrix(n, m);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < n; ++i) taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(n);
  // Create Task
  auto testTaskSequential = std::make_shared<morozov_e_min_val_in_rows_matrix::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(-1, res[i]);
  }
}
