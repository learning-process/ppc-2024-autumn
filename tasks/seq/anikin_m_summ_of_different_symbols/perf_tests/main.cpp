// Copyright 2024 Anikin Maksim
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/anikin_m_summ_of_different_symbols/include/ops_seq.hpp"

TEST(SumDifSymSequential_perf_test, test_pipeline_run) {
  // Create data
  char str1[] = "abcd";
  char str2[] = "acdb"

  std::vector<std::string> in{str1, str2};
  std::vector<int> out(1, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<nesterov_a_test_task_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(3, out[0]);
}

TEST(SumDifSymSequential_perf_test, test_task_run) {
  // Create data
  char str1[] = "abcd";
  char str2[] = "acdb"

  std::vector<std::string> in{str1, str2};
  std::vector<int> out(1, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<nesterov_a_test_task_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(count, out[0]);
}
