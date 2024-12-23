// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_seq.hpp"
static std::vector<double> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  std::vector<double> vec(sz);
  for (int i = 0; i < sz; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, test_pipeline_run) {
  int N = 400;
  int block_size = 1;
  std::vector<double> matrixA = getRandomVector(N * N);
  std::vector<double> matrixB = getRandomVector(N * N);
  std::vector<double> matrixC(N * N, 0.0);
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
  std::vector<double> matrixA = getRandomVector(N * N);
  std::vector<double> matrixB = getRandomVector(N * N);
  std::vector<double> matrixC(N * N, 0.0);
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
