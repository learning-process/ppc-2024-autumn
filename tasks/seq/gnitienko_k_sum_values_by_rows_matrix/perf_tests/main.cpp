// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/gnitienko_k_sum_values_by_rows_matrix/include/ops_seq.hpp"

TEST(gnitienko_k_sum_row_seq, test_pipeline_run) {
  const int rows = 3000;
  const int cols = 3000;

  // Create data
  std::vector<int> in(rows * cols, 0);
  for (int i = 0; i < rows; ++i) {
    in[i * cols] = i;
  }
  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      expect[i] += in[i * cols + j];
    }
  }
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(rows));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(cols));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(out.size()));

  // Create Task
  auto testTaskSequential = std::make_shared<gnitienko_k_sum_row_seq::SumByRowSeq>(taskDataSeq);

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
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expect, out);
}

TEST(gnitienko_k_sum_row_seq, test_task_run) {
  const int rows = 3000;
  const int cols = 3000;

  // Create data
  std::vector<int> in(rows * cols, 0);
  for (int i = 0; i < rows; ++i) {
    in[i * cols] = i;
  }
  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      expect[i] += in[i * cols + j];
    }
  }
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(rows));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(cols));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(out.size()));

  // Create Task
  auto testTaskSequential = std::make_shared<gnitienko_k_sum_row_seq::SumByRowSeq>(taskDataSeq);

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
  ASSERT_EQ(expect, out);
}
