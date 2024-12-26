// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kolodkin_g_hoar_merge_sort/include/ops_seq.hpp"

TEST(seq_kolodkin_g_hoar_merge_sort_test, test_pipeline_run) {
  // Create data
  std::vector<int> vec;
  std::vector<int> out(10000);
  for (unsigned i = 0; i < 10000; i++) {
    vec.push_back(-1000 + rand() % 5000);
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskDataSeq->inputs_count.emplace_back(vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kolodkin_g_hoar_merge_sort_seq::TestTaskSequential>(taskDataSeq);

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
  std::sort(vec.begin(), vec.end());
  out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
  for (unsigned i = 0; i < vec.size(); i++) {
    ASSERT_EQ(vec[i], out[i]);
  }
}

TEST(seq_kolodkin_g_hoar_merge_sort_test, test_task_run) {
  // Create data
  std::vector<int> vec;
  std::vector<int> out(10000);
  for (unsigned i = 0; i < 10000; i++) {
    vec.push_back(-1000 + rand() % 5000);
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
  taskDataSeq->inputs_count.emplace_back(vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kolodkin_g_hoar_merge_sort_seq::TestTaskSequential>(taskDataSeq);

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
  std::sort(vec.begin(), vec.end());
  out = *reinterpret_cast<std::vector<int> *>(taskDataSeq->outputs[0]);
  for (unsigned i = 0; i < vec.size(); i++) {
    ASSERT_EQ(vec[i], out[i]);
  }
}
