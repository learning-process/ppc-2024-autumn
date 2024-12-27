// Copyright 2024 Korobeinikov Arseny
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B/include/ops_seq_korobeinikov.hpp"

TEST(sequential_korobeinikov_perf_test_lab_02, test_pipeline_run) {
  // Create data
  std::vector<int> A(53361, 1);
  std::vector<int> B(53361, 1);
  int count_rows_A = 231;
  int count_cols_A = 231;
  int count_rows_B = 231;
  int count_cols_B = 231;

  std::vector<int> out(53361, 0);
  std::vector<int> right_answer(53361, 231);
  int count_rows_out = 0;
  int count_cols_out = 0;
  int count_rows_RA = 231;
  int count_cols_RA = 231;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out));
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  auto testTaskSequential = std::make_shared<korobeinikov_a_test_task_seq_lab_02::TestTaskSequential>(taskDataSeq);

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
  for (size_t i = 0; i < right_answer.size(); i++) {
    ASSERT_EQ(right_answer[i], out[i]);
  }
  ASSERT_EQ(count_rows_out, count_rows_RA);
  ASSERT_EQ(count_cols_out, count_cols_RA);
}

TEST(sequential_korobeinikov_perf_test_lab_02, test_task_run) {
  // Create data
  std::vector<int> A(53361, 1);
  std::vector<int> B(53361, 1);
  int count_rows_A = 231;
  int count_cols_A = 231;
  int count_rows_B = 231;
  int count_cols_B = 231;

  std::vector<int> out(53361, 0);
  std::vector<int> right_answer(53361, 231);
  int count_rows_out = 0;
  int count_cols_out = 0;
  int count_rows_RA = 231;
  int count_cols_RA = 231;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out));
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  auto testTaskSequential = std::make_shared<korobeinikov_a_test_task_seq_lab_02::TestTaskSequential>(taskDataSeq);

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
  for (size_t i = 0; i < right_answer.size(); i++) {
    ASSERT_EQ(right_answer[i], out[i]);
  }
  ASSERT_EQ(count_rows_out, count_rows_RA);
  ASSERT_EQ(count_cols_out, count_cols_RA);
}
