// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_seq.hpp"
static void generateMatrix(size_t num_rows, size_t num_cols, std::vector<double> &matrix) {
  double min_val = -100.0;
  double max_val = 100.0;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<float> dist(min_val, max_val);
  matrix.resize(num_rows * num_cols);
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      matrix[i * num_cols + j] = dist(gen);
    }
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, test_pipeline_run) {
  int N = 400;
  int block_size = 1;
  std::vector<double> matrixA(N * N, 0.0);
  std::vector<double> matrixB(N * N, 0.0);
  std::vector<double> matrixC(N * N, 0.0);
  generateMatrix(N, N, matrixA);
  generateMatrix(N, N, matrixB);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrixC.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  auto testTaskSequential =
      std::make_shared<lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential>(taskDataSeq);
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
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, test_task_run) {
  int N = 400;
  int block_size = 1;
  std::vector<double> matrixA(N * N, 0.0);
  std::vector<double> matrixB(N * N, 0.0);
  std::vector<double> matrixC(N * N, 0.0);
  generateMatrix(N, N, matrixA);
  generateMatrix(N, N, matrixB);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrixC.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  auto testTaskSequential =
      std::make_shared<lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential>(taskDataSeq);
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
}
