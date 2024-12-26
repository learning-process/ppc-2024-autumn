#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

std::vector<double> generateRandomMatrix(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0);
  std::vector<double> matrix(size * size);
  for (int i = 0; i < size * size; i++) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

TEST(nasedkin_e_strassen_algorithm_perf_test, test_pipeline_run) {
  int matrixSize = 1024;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA.size());
  taskDataSeq->inputs_count.emplace_back(matrixB.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resultSeq.size());

  auto testSeqTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ>(taskDataSeq);
  ASSERT_EQ(testSeqTask->validation(), true);
  testSeqTask->pre_processing();
  testSeqTask->run();
  testSeqTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSeqTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(nasedkin_e_strassen_algorithm_perf_test, test_task_run) {
  int matrixSize = 1024;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA.size());
  taskDataSeq->inputs_count.emplace_back(matrixB.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resultSeq.size());

  auto testSeqTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ>(taskDataSeq);
  ASSERT_EQ(testSeqTask->validation(), true);
  testSeqTask->pre_processing();
  testSeqTask->run();
  testSeqTask->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSeqTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
}