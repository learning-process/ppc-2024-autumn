// Copyright 2024 Korobeinikov Arseny
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/korobeinikov_dijkstras_algorithm/include/ops_seq_korobeinikov.hpp"

TEST(korobeinikov_sequential_perf_test_lab_03, test_pipeline_run) {
  // Create data
  size_t size = 10000;

  std::vector<int> values(size * size, 1);
  std::vector<int> col(size * size, 1);
  std::vector<int> RowIndex(size + 1, 1);

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      col[i * size + j] = j;
    }
  }
  for (size_t i = 0; i < size + 1; i++) {
    RowIndex[i] = i * size;
  }

  int sv = 0;

  std::vector<int> out(size, 0);
  std::vector<int> right_answer(size, 1);
  right_answer[0] = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
  taskDataSeq->inputs_count.emplace_back(col.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
  taskDataSeq->inputs_count.emplace_back(RowIndex.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<korobeinikov_a_test_task_seq_lab_03::TestTaskSequential>(taskDataSeq);

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
}

TEST(korobeinikov_sequential_perf_test_lab_03, test_task_run) {
  // Create data
  size_t size = 10000;

  std::vector<int> values(size * size, 1);
  std::vector<int> col(size * size, 1);
  std::vector<int> RowIndex(size + 1, 1);

  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      col[i * size + j] = j;
    }
  }
  for (size_t i = 0; i < size + 1; i++) {
    RowIndex[i] = i * size;
  }

  int sv = 0;

  std::vector<int> out(size, 0);
  std::vector<int> right_answer(size, 1);
  right_answer[0] = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
  taskDataSeq->inputs_count.emplace_back(col.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
  taskDataSeq->inputs_count.emplace_back(RowIndex.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<korobeinikov_a_test_task_seq_lab_03::TestTaskSequential>(taskDataSeq);

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
}