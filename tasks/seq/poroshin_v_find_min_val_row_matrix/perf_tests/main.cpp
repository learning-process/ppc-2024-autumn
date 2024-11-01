// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/poroshin_v_find_min_val_row_matrix/include/ops_seq.hpp"

TEST(poroshin_v_find_min_val_row_matrix_seq, test_pipeline_run) {
  // Create data

  const int n = 5000, m = 5000;

  // Create TaskData

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Create Task

  auto testTaskSequential = std::make_shared<poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential>(taskDataSeq);

  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  std::vector<int> result(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(m);
 // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;  // Set the number of runs as needed
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

  for (int i = 0; i < m; i++) {
    ASSERT_EQ(result[i], INT_MIN);
  }
}

TEST(poroshin_v_find_min_val_row_matrix_seq, test_task_run) {
  // Create data

  const int n = 5000, m = 5000;

  // Create TaskData

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  // Create Task

  auto testTaskSequential = std::make_shared<poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential>(taskDataSeq);

  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  std::vector<int> result(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(m);
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

  for (int i = 0; i < m; i++) {
    ASSERT_EQ(result[i], INT_MIN);
  }
}
