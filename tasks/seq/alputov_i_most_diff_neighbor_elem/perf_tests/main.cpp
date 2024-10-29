// Copyright 2024 Alputov Ivan
#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/alputov_i_most_diff_neighbor_elem/include/ops_seq.hpp"

// Test for pipeline run performance of the MostDiffNeighborElemSeq task
TEST(alputov_i_most_diff_neighbor_elem_perf_test, test_pipeline_run) {
  const int count = 10000000;  // Увеличен размер вектора

  // Create input data: a vector with a sequence of increasing integers
  std::vector<int> in(count);
  for (int i = 0; i < count; ++i) {
    in[i] = i;  // Fill with increasing values for difference testing
  }
  std::vector<std::pair<int, int>> out(1, {0, 0});  // Output to hold max difference pair

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<alputov_i_most_diff_neighbor_elem_seq::MostDiffNeighborElemSeq>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;  // Number of times the task will be run
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;  // Return elapsed time in seconds
  };

  // Create and initialize performance results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Изменена проверка результата
  int expectedMaxDifference = std::abs(in.back() - in[in.size() - 2]);  // expected max difference adjusted
  ASSERT_EQ(std::abs(out[0].second - out[0].first), expectedMaxDifference);
}

// Test for direct task run performance of the MostDiffNeighborElemSeq task
TEST(alputov_i_most_diff_neighbor_elem_perf_test, test_task_run) {
  const int count = 10000000;  // Увеличен размер вектора

  // Create input data: a vector with a sequence of increasing integers
  std::vector<int> in(count);
  for (int i = 0; i < count; ++i) {
    in[i] = i;  // Fill with increasing values for difference testing
  }
  std::vector<std::pair<int, int>> out(1, {0, 0});  // Output to hold max difference pair

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<alputov_i_most_diff_neighbor_elem_seq::MostDiffNeighborElemSeq>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;  // Number of times the task will be run
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;  // Return elapsed time in seconds
  };

  // Create and initialize performance results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Изменена проверка результата
  int expectedMaxDifference = std::abs(in.back() - in[in.size() - 2]);  // expected max difference adjusted
  ASSERT_EQ(std::abs(out[0].second - out[0].first), expectedMaxDifference);
}
