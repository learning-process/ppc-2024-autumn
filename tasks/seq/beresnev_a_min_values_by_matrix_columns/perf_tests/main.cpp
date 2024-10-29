// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>
#include <ctime>
#include <cstdlib>

#include "core/perf/include/perf.hpp"
#include "seq/beresnev_a_min_values_by_matrix_columns/include/ops_seq.hpp"

TEST(sequential_beresnev_a_min_values_by_matrix_columns_perf_test, test_pipeline_run) {
  const std::uint32_t N = 2000;  // Количество строк
  const std::uint32_t M = 10000;  // Количество столбцов

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  std::vector<int> in(N * M);
  for (int i = 0; i < N * M; ++i) {
    in[i] = std::rand() % 2000 - 1000;
  }

  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential>(taskDataSeq);
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

  for (std::uint32_t i = 0; i < M; ++i) {
    int expectedMin = in[i];
    for (std::uint32_t j = 1; j < N; ++j) {
      int currentValue = in[j * M + i];
      if (currentValue < expectedMin) {
        expectedMin = currentValue;
      }
    }
    ASSERT_EQ(out[i], expectedMin);
  }
}

TEST(sequential_beresnev_a_min_values_by_matrix_columns_perf_test, test_task_run) {
  const std::uint32_t N = 2000;  // Количество строк
  const std::uint32_t M = 10000;  // Количество столбцов

  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  std::vector<int> in(N * M);
  for (int i = 0; i < N * M; ++i) {
    in[i] = std::rand() % 2000 - 1000;
  }

  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskDataSeq->inputs_count.emplace_back(n.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
  taskDataSeq->inputs_count.emplace_back(m.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<beresnev_a_min_values_by_matrix_columns_seq::TestTaskSequential>(taskDataSeq);

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
  
  for (std::uint32_t i = 0; i < M; ++i) {
    int expectedMin = in[i];
    for (std::uint32_t j = 1; j < N; ++j) {
      int currentValue = in[j * M + i];
      if (currentValue < expectedMin) {
        expectedMin = currentValue;
      }
    }
    ASSERT_EQ(out[i], expectedMin);
  }
}