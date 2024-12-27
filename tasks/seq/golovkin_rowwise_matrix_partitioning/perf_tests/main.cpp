// Golovkin Maksim Task#2
#include <gtest/gtest.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/golovkin_rowwise_matrix_partitioning/include/ops_seq.hpp"

using namespace golovkin_rowwise_matrix_partitioning;
using ppc::core::Perf;
using ppc::core::TaskData;

TEST(golovkin_rowwise_matrix_partitioning, test_pipeline_run) {
  const int N = 500;

  std::vector<std::vector<double>> A(N, std::vector<double>(N, 1.0));
  std::vector<std::vector<double>> B(N, std::vector<double>(N, 1.0));
  std::vector<std::vector<double>> result(N, std::vector<double>(N, 0.0));

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(&B));
  taskDataSeq->inputs_count = {sizeof(std::vector<std::vector<double>>), sizeof(std::vector<std::vector<double>>)};

  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskDataSeq->outputs_count.push_back(sizeof(std::vector<std::vector<double>>));

  auto matrixMultiplicationTask = std::make_shared<MatrixMultiplicationTask>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(matrixMultiplicationTask);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(result.size(), static_cast<size_t>(N));
  ASSERT_EQ(result[0].size(), static_cast<size_t>(N));
}

TEST(golovkin_rowwise_matrix_partitioning, test_task_run) {
  const int N = 500;

  std::vector<std::vector<double>> A(N, std::vector<double>(N, 1.0));
  std::vector<std::vector<double>> B(N, std::vector<double>(N, 1.0));
  std::vector<std::vector<double>> result(N, std::vector<double>(N, 0.0));

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(&A));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(&B));
  taskDataSeq->inputs_count = {sizeof(std::vector<std::vector<double>>), sizeof(std::vector<std::vector<double>>)};

  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  taskDataSeq->outputs_count.push_back(sizeof(std::vector<std::vector<double>>));

  auto matrixMultiplicationTask = std::make_shared<MatrixMultiplicationTask>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(matrixMultiplicationTask);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(result.size(), static_cast<size_t>(N));
  ASSERT_EQ(result[0].size(), static_cast<size_t>(N));
}