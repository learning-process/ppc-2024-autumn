// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/borisov_s_sum_of_rows/include/ops_seq.hpp"

TEST(borisov_s_sum_of_rows, test_pipeline_run) {
  size_t rows = 5000;
  size_t cols = 5000;

  std::vector<int> matrix(rows * cols, 1);
  std::vector<int> row_sums(rows, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.push_back(row_sums.size());

  auto sumOfRowsTask = std::make_shared<borisov_s_sum_of_rows::SumOfRowsTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sumOfRowsTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < rows; i++) {
    ASSERT_EQ(row_sums[i], 5000);
  }
}

TEST(borisov_s_sum_of_rows, test_task_run) {
  size_t rows = 5000;
  size_t cols = 5000;

  std::vector<int> matrix(rows * cols, 1);
  std::vector<int> row_sums(rows, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.push_back(row_sums.size());

  auto sumOfRowsTask = std::make_shared<borisov_s_sum_of_rows::SumOfRowsTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sumOfRowsTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < rows; i++) {
    ASSERT_EQ(row_sums[i], 5000);
  }
}
