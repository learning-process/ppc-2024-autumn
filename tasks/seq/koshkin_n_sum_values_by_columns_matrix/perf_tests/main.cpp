// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/koshkin_n_sum_values_by_columns_matrix/include/ops_seq.hpp"

TEST(koshkin_n_sum_values_by_columns_matrix_seq, test_pipeline_run) {
  int rows = 3000;
  int columns = 3000;

  std::vector<int> matrix(columns * rows, 0);
  std::vector<int> res_out(columns, 0);
  std::vector<int> exp_res(columns, 0);
  for (int i = 0; i < 1000; i += 2) {
    matrix[i] = 1;
    exp_res[i] = 1;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  // Create Task
  auto testTaskSequential =
      std::make_shared<koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential>(taskDataSeq);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));
  taskDataSeq->outputs_count.emplace_back(res_out.size());

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

  ASSERT_EQ(res_out, exp_res);
}

TEST(koshkin_n_sum_values_by_columns_matrix_seq, test_task_run) {
  int rows = 3000;
  int columns = 3000;

  std::vector<int> matrix(columns * rows, 0);
  std::vector<int> res_out(columns, 0);
  std::vector<int> exp_res(columns, 0);
  for (int i = 0; i < 1000; i += 2) {
    matrix[i] = 1;
    exp_res[i] = 1;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto testTaskSequential =
      std::make_shared<koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential>(taskDataSeq);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));
  taskDataSeq->outputs_count.emplace_back(res_out.size());

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
  ASSERT_EQ(res_out, exp_res);
}