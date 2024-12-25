// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kolokolova_d_radix_integer_merge_sort/include/ops_seq.hpp"

TEST(kolokolova_d_radix_integer_merge_sort_seq, test_pipeline_run) {
  std::vector<int> unsorted_vector(120000, 0);

  std::vector<int> sorted_vector(int(unsorted_vector.size()), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
  taskDataSeq->inputs_count.emplace_back(unsorted_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_vector.data()));
  taskDataSeq->outputs_count.emplace_back(sorted_vector.size());

  auto testTaskSequential =
      std::make_shared<kolokolova_d_radix_integer_merge_sort_seq::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < sorted_vector.size(); i++) {
    EXPECT_EQ(sorted_vector[i], 0);
  }
}

TEST(kolokolova_d_radix_integer_merge_sort_seq, test_task_run) {
  std::vector<int> unsorted_vector(120000, 1);

  std::vector<int> sorted_vector(int(unsorted_vector.size()), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(unsorted_vector.data()));
  taskDataSeq->inputs_count.emplace_back(unsorted_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_vector.data()));
  taskDataSeq->outputs_count.emplace_back(sorted_vector.size());

  auto testTaskSequential =
      std::make_shared<kolokolova_d_radix_integer_merge_sort_seq::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  for (size_t i = 0; i < sorted_vector.size(); i++) {
    EXPECT_EQ(1, sorted_vector[i]);
  }
}