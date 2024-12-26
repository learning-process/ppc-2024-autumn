// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/lysov_i_simple_iteration_method/include/ops_seq.hpp"

void generate_diagonally_dominant_matrix(int size, std::vector<double> &matrix, std::vector<double> &b) {
  matrix.resize(size * size);
  b.resize(size);

  for (int i = 0; i < size; ++i) {
    double sum = 0.0;
    matrix[i * size + i] = 2.0 * size;

    for (int j = 0; j < size; ++j) {
      if (i != j) {
        matrix[i * size + j] = 1.0;
        sum += std::abs(matrix[i * size + j]);
      }
    }

    b[i] = matrix[i * size + i] * 1.0 + sum;
  }
}

TEST(lysov_i_simple_iteration_method_seq, test_pipeline_run) {
  const int input_size = 1000;

  std::vector<double> matrix;
  std::vector<double> b;
  generate_diagonally_dominant_matrix(input_size, matrix, b);

  std::vector<double> x(input_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->outputs_count.push_back(input_size);

  auto testTaskSequential = std::make_shared<lysov_i_simple_iteration_method_seq::SlaeIterationTask>(taskDataSeq);

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

  std::vector<double> expected_solution(input_size, 1.0);

  double tolerance = 1e-6;
  for (int i = 0; i < input_size; ++i) {
    ASSERT_NEAR(x[i], expected_solution[i], tolerance);
  }
}

TEST(lysov_i_simple_iteration_method_seq, test_task_run) {
  const int input_size = 1000;

  std::vector<double> matrix;
  std::vector<double> b;
  generate_diagonally_dominant_matrix(input_size, matrix, b);

  std::vector<double> x(input_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->outputs_count.push_back(input_size);

  auto testTaskSequential = std::make_shared<lysov_i_simple_iteration_method_seq::SlaeIterationTask>(taskDataSeq);

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

  std::vector<double> expected_solution(input_size, 1.0);

  double tolerance = 1e-6;
  for (int i = 0; i < input_size; ++i) {
    ASSERT_NEAR(x[i], expected_solution[i], tolerance);
  }
}