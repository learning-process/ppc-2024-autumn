// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/tsatsyn_a_vector_dot_product/include/ops_seq.hpp"

std::vector<int> toGetRandomVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> v(size);
  for (int i = 0; i < size; i++) {
    v[i] = gen() % 200 + gen() % 10;
  }
  return v;
}

TEST(sequential_tsatsyn_a_vector_dot_product_perf_test, test_pipeline_run) {
  const int size = 10000000;

  // Create data
  std::vector<int> v1 = toGetRandomVector(size);
  std::vector<int> v2 = toGetRandomVector(size);
  std::vector<int> ans(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
  taskDataSeq->inputs_count.emplace_back(v2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
  taskDataSeq->outputs_count.emplace_back(ans.size());

  // Create Task
  auto testTaskSequential = std::make_shared<tsatsyn_a_vector_dot_product_seq::TestTaskSequential>(taskDataSeq);

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

  // ASSERT_EQ(tsatsyn_a_vector_dot_product::resulting(v1, v2), ans[0]);
  ASSERT_EQ(tsatsyn_a_vector_dot_product_seq::resulting(v1, v2), ans[0]);
}

TEST(sequential_tsatsyn_a_vector_dot_product_perf_test, test_task_run) {
  const int size = 10000000;

  // Create data
  std::vector<int> v1 = toGetRandomVector(size);
  std::vector<int> v2 = toGetRandomVector(size);
  std::vector<int> ans(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));
  taskDataSeq->inputs_count.emplace_back(v2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans.data()));
  taskDataSeq->outputs_count.emplace_back(ans.size());

  // Create Task
  auto testTaskSequential = std::make_shared<tsatsyn_a_vector_dot_product_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(tsatsyn_a_vector_dot_product_seq::resulting(v1, v2), ans[0]);
}