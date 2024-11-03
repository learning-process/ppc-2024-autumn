// Copyright 2024 Korobeinikov Arseny
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/korobeinikov_a_max_elements_in_rows_of_matrix/include/ops_seq_korobeinikov.hpp"

TEST(sequential_korobeinikov_perf_test, test_pipeline_run) {
  // Create data
  int count_rows = 500;  // not const, because reinterpret_cast does not work with const
  std::vector<int> matrix(count_rows * 10000, 10);

  std::vector<int> seq_res(count_rows, 0);
  std::vector<int> right_answer = std::vector<int>(count_rows, 10);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  auto testTaskSequential = std::make_shared<korobeinikov_a_test_task_seq::TestTaskSequential>(taskDataSeq);

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
  for (unsigned i = 0; i < seq_res.size(); i++) {
    EXPECT_EQ(10, seq_res[0]);
  }
}

TEST(sequential_korobeinikov_perf_test, test_task_run) {
  // Create data
  int count_rows = 500;  // not const, because reinterpret_cast does not work with const
  std::vector<int> matrix(count_rows * 100000, 10);

  std::vector<int> seq_res(count_rows, 0);
  std::vector<int> right_answer = std::vector<int>(count_rows, 10);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  auto testTaskSequential = std::make_shared<korobeinikov_a_test_task_seq::TestTaskSequential>(taskDataSeq);

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
  for (unsigned i = 0; i < seq_res.size(); i++) {
    EXPECT_EQ(10, seq_res[0]);
  }
}
