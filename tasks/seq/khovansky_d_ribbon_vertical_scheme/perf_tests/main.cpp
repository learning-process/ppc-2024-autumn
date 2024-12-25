// Copyright 2024 Khovansky Dmitry
#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/khovansky_d_ribbon_vertical_scheme/include/ops_seq.hpp"

TEST(khovansky_d_ribbon_vertical_scheme_seq, test_pipeline_run) {
  std::vector<int> input_matrix;
  std::vector<int> input_vector;
  std::vector<int> output_vector;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rows_count;
  int columns_count;

  rows_count = 8192;
  columns_count = 8192;

  input_vector.resize(columns_count);
  input_matrix.resize(rows_count * columns_count);
  output_vector.resize(columns_count, 0);

  for (int i = 0; i < rows_count * columns_count; i++) {
    input_matrix[i] = (rand() % 1000) - 500;
    if (i < rows_count) {
      input_vector[i] = (rand() % 1000) - 500;
    }
  }
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  taskDataSeq->outputs_count.emplace_back(output_vector.size());

  auto testTaskSequential =
      std::make_shared<khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq>(taskDataSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<int> expected_result(columns_count, 0);

  for (int i = 0; i < rows_count; i++) {
    for (int j = 0; j < columns_count; j++) {
      expected_result[j] += input_matrix[i * columns_count + j] * input_vector[i];
    }
  }

  ASSERT_EQ(output_vector, expected_result);
}

TEST(khovansky_d_ribbon_vertical_scheme_seq, test_task_run) {
  std::vector<int> input_matrix;
  std::vector<int> input_vector;
  std::vector<int> output_vector;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rows_count;
  int columns_count;

  rows_count = 8192;
  columns_count = 8192;

  input_vector.resize(columns_count);
  input_matrix.resize(rows_count * columns_count);
  output_vector.resize(columns_count, 0);

  for (int i = 0; i < rows_count * columns_count; i++) {
    input_matrix[i] = (rand() % 1000) - 500;
    if (i < rows_count) {
      input_vector[i] = (rand() % 1000) - 500;
    }
  }
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  taskDataSeq->inputs_count.emplace_back(input_vector.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  taskDataSeq->outputs_count.emplace_back(output_vector.size());

  auto testTaskSequential =
      std::make_shared<khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq>(taskDataSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<int> expected_result(columns_count, 0);

  for (int i = 0; i < rows_count; i++) {
    for (int j = 0; j < columns_count; j++) {
      expected_result[j] += input_matrix[i * columns_count + j] * input_vector[i];
    }
  }

  ASSERT_EQ(output_vector, expected_result);
}