// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shishkarev_a_sum_of_vector_elements/include/ops_seq.hpp"

TEST(shishkarev_a_sum_of_vector_elements_seq, test_pipeline_run) {
  const int count = 10000000;

  // Create data
  std::vector<int> in(count, 1);  // Populate with 1 to simulate a valid sum
  std::vector<int> out(1, 0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto vectorSumSequential =
      std::make_shared<shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int>>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;  // Adjusted for single-threaded testing
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and initialize perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(vectorSumSequential);

  // Run pipeline and verify output
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  const int expected_sum = std::accumulate(in.begin(), in.end(), 0);
  ASSERT_EQ(out[0], expected_sum);
}

TEST(shishkarev_a_sum_of_vector_elements_seq, test_task_run) {
  const int count = 10000000;

  // Create data
  std::vector<int> in(count, 1);  // Populate with 1 to simulate a valid sum
  std::vector<int> out(1, 0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto vectorSumSequential =
      std::make_shared<shishkarev_a_sum_of_vector_elements_seq::VectorSumSequential<int>>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1;  // Adjusted for single-threaded testing
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and initialize perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(vectorSumSequential);

  // Run task and verify output
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  const int expected_sum = std::accumulate(in.begin(), in.end(), 0);
  ASSERT_EQ(out[0], expected_sum);
}
