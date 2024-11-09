#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/varfolomeev_g_matrix_max_rows_vals/include/ops_seq.hpp"

std::vector<std::vector<int>> generateMatrix(int rows, int cols, int a, int b) {
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

int searchMaxInVec(std::vector<int> vec) {
  int max = vec[0];
  for (size_t i = 1; i < vec.size(); i++) {
    if (max < vec[i]) max = vec[i];
  }
  return max;
}

TEST(sequential_varfolomeev_g_matrix_max_rows_perf_test, test_pipeline_run_10000x10000) {
  int rows = 10000;
  int cols = 10000;
  int a = -100;
  int b = 100;

  // Create data
  std::vector<std::vector<int>> in = generateMatrix(rows, cols, a, b);
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < rows; ++i) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto maxInRows = std::make_shared<varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(maxInRows);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Check results
  for (int i = 0; i < rows; ++i) {
    int expected_max = searchMaxInVec(in[i]);
    ASSERT_EQ(out[i], expected_max);
  }
}

TEST(sequential_varfolomeev_g_matrix_max_rows_perf_test, test_task_run_10000x10000) {
  const int rows = 10000;
  const int cols = 10000;
  const int a = -100;
  const int b = 100;

  // Create data
  std::vector<std::vector<int>> in = generateMatrix(rows, cols, a, b);
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < rows; ++i) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto maxInRows = std::make_shared<varfolomeev_g_matrix_max_rows_vals_seq::MaxInRows>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(maxInRows);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Check results
  for (int i = 0; i < rows; ++i) {
    int expected_max = searchMaxInVec(in[i]);
    ASSERT_EQ(out[i], expected_max);
  }
}