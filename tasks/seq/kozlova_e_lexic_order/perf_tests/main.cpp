// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kozlova_e_lexic_order/include/ops_seq.hpp"

TEST(kozlova_e_lexic_order_perf_test, test_pipeline_run) {
  char *str1 = new char[1000000 + 1];
  char *str2 = new char[1000000 + 1];

  for (int i = 0; i < 30; i++) {
    str1[i] = 'a';
    str2[i] = 'b';
  }
  for (int i = 30; i < 60; i++) {
    str1[i] = 'C';
    str2[i] = 'd';
  }
  for (int i = 60; i < 1000000; i++) {
    str1[i] = 'e';
    str2[i] = 'f';
  }
  str1[1000000] = '\0';
  str2[1000000] = '\0';

  std::vector<int> out(2, 0);
  std::vector<int> expect(2, 1);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(2));

  // Create Task
  auto testTaskSequential = std::make_shared<kozlova_e_lexic_order::StringComparator>(taskDataSeq);

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
  delete[] str1;
  delete[] str2;
}

TEST(kozlova_e_lexic_order_perf_test, test_task_run) {
  char *str1 = new char[1000000 + 1];
  char *str2 = new char[1000000 + 1];

  for (int i = 0; i < 30; i++) {
    str1[i] = 'a';
    str2[i] = 'b';
  }
  for (int i = 30; i < 60; i++) {
    str1[i] = 'C';
    str2[i] = 'd';
  }
  for (int i = 60; i < 1000000; i++) {
    str1[i] = 'e';
    str2[i] = 'f';
  }
  str1[1000000] = '\0';
  str2[1000000] = '\0';

  std::vector<int> out(2, 0);
  std::vector<int> expect(2, 1);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(2));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(2));

  // Create Task
  auto testTaskSequential = std::make_shared<kozlova_e_lexic_order::StringComparator>(taskDataSeq);

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
  delete[] str1;
  delete[] str2;
}