// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/zaitsev_a_min_of_vector_elements/include/ops_seq.hpp"

TEST(zaitsev_a_min_of_vector_elements_sequentional, test_pipeline_run) {
  const int length = 10e6;
  const int extrema = -105;
  const int minRangeValue = -100;
  const int maxRangeValue = 1000;

  std::mt19937 gen(31415);

  // Create data
  std::vector<int> in(length);
  for (size_t i = 0; i < length; i++) {
    int j = minRangeValue + gen() % (maxRangeValue - minRangeValue + 1);
    in[i] = j;
  }
  in[length / 2] = extrema;

  std::vector<int> out(1, extrema);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<zaitsev_a_min_of_vector_elements_seq::MinOfVectorElementsSequential>(taskDataSeq);

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
  ASSERT_EQ(extrema, out[0]);
}

TEST(zaitsev_a_min_of_vector_elements_sequentional, test_task_run) {
  const int length = 10e6;
  const int extrema = -105;
  const int minRangeValue = -100;
  const int maxRangeValue = 1000;

  std::mt19937 gen(31415);

  // Create data
  std::vector<int> in(length);
  for (size_t i = 0; i < length; i++) {
    int j = minRangeValue + gen() % (maxRangeValue - minRangeValue + 1);
    in[i] = j;
  }
  in[length / 2] = extrema;

  std::vector<int> out(1, extrema);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<zaitsev_a_min_of_vector_elements_seq::MinOfVectorElementsSequential>(taskDataSeq);

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
  ASSERT_EQ(extrema, out[0]);
}
