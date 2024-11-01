#include <gtest/gtest.h>

#include <iostream>
#include <thread>

#include "core/perf/include/perf.hpp"
#include "seq/sadikov_I_sum_values_by_columns_matrix/include/sq_task.h"

TEST(Sadikov_sequential_perf_test, matrix_test_pipeline_run) {
  int rows_count = 3000;
  int columns_count = 3000;
  std::vector<int> in(rows_count * columns_count, 1);
  std::vector<int> in_index{rows_count, columns_count};
  std::vector<int> out(columns_count, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);

  auto testTaskSequential = std::make_shared<sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(in_index[1], out[0]);
}

TEST(Sadikov_sequential_perf_test, matrix_test_run) {
  int rows_count = 3000;
  int columns_count = 3000;
  std::vector<int> in(rows_count * columns_count, 1);
  std::vector<int> in_index{rows_count, columns_count};
  std::vector<int> out(columns_count, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);

  auto testTaskSequential = std::make_shared<sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(in_index[1], out[0]);
}