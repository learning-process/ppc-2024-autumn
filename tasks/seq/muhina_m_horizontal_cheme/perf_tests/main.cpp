// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/muhina_m_horizontal_cheme/include/ops_seq.hpp"

TEST(muhina_m_horizontal_cheme, test_pipeline_run) {
  int size = 1000;
  std::vector<int> matrix(size * size, 1);
  std::vector<int> vec(size, 1);
  std::vector<int> out(size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
  taskDataSeq->inputs_count.emplace_back(vec.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto HorizontalSchemeSequential =
      std::make_shared<muhina_m_horizontal_cheme_seq::HorizontalSchemeSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(HorizontalSchemeSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  int expected_result = size;
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(out[i], expected_result);
  }
}

TEST(muhina_m_horizontal_cheme, test_task_run) {
  int size = 1000;
  std::vector<int> matrix(size * size, 1);
  std::vector<int> vec(size, 1);
  std::vector<int> out(size, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
  taskDataSeq->inputs_count.emplace_back(vec.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto HorizontalSchemeSequential =
      std::make_shared<muhina_m_horizontal_cheme_seq::HorizontalSchemeSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(HorizontalSchemeSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  int expected_result = size;
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(out[i], expected_result);
  }
}
