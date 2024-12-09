// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/frolova_e_matrix_multiplication/include/ops_seq.hpp"

TEST(frolova_e_matrix_multiplication_seq, test_pipeline_run) {

    // Create data
  std::vector<int> values_1 = {100, 100};
  std::vector<int> values_2 = {100, 100};
  std::vector<int> matrixA_;
  frolova_e_matrix_multiplication_seq::randomNumVec(10000, matrixA_);
  std::vector<int> matrixB_;
  frolova_e_matrix_multiplication_seq::randomNumVec(10000, matrixB_);
  std::vector<int> out(10000);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixB_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<frolova_e_matrix_multiplication_seq::matrixMultiplication>(taskDataSeq);

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

  std::vector<int> count = frolova_e_matrix_multiplication_seq::Multiplication(100, 100, 100, matrixA_, matrixB_);

  ASSERT_EQ(count, out);
}

TEST(frolova_e_matrix_multiplication_seq, test_task_run) {
  // Create data
  std::vector<int> values_1 = {100, 100};
  std::vector<int> values_2 = {100, 100};
  std::vector<int> matrixA_;
  frolova_e_matrix_multiplication_seq::randomNumVec(10000, matrixA_);
  std::vector<int> matrixB_;
  frolova_e_matrix_multiplication_seq::randomNumVec(10000, matrixB_);
  std::vector<int> out(10000);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixB_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<frolova_e_matrix_multiplication_seq::matrixMultiplication>(taskDataSeq);

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
  std::vector<int> count = frolova_e_matrix_multiplication_seq::Multiplication(100, 100, 100, matrixA_, matrixB_);
  ASSERT_EQ(count, out);
}
