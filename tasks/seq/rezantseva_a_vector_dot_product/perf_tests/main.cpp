// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/rezantseva_a_vector_dot_product/include/ops_seq.hpp"

TEST(rezantseva_a_vector_dot_product_seq, test_pipeline_run) {
  const int count = 100000000;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs_count.emplace_back(v2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<rezantseva_a_vector_dot_product_seq::TestTaskSequential>(taskDataSeq);

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

  int answer = rezantseva_a_vector_dot_product_seq::vectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}

TEST(rezantseva_a_vector_dot_product_seq, test_task_run) {
  const int count = 100000000;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs_count.emplace_back(v2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<rezantseva_a_vector_dot_product_seq::TestTaskSequential>(taskDataSeq);
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

  int answer = rezantseva_a_vector_dot_product_seq::vectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}