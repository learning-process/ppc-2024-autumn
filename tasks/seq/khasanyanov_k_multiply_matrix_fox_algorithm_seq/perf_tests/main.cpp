#include <gtest/gtest.h>

#include <memory>

#include "../include/matrix_operations.hpp"
#include "core/perf/include/perf.hpp"

using namespace khasanyanov_k_fox_algorithm;

template <typename DataType>
static std::shared_ptr<ppc::core::TaskData> create_task_data(matrix<DataType>& A, matrix<DataType>& B,
                                                             matrix<DataType>& C) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(*A));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(A.rows));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(A.columns));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(A.size()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(*B));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(B.rows));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(B.columns));
  taskData->inputs_count.emplace_back(static_cast<uint32_t>(B.size()));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(*C));
  taskData->outputs_count.emplace_back(static_cast<uint32_t>(C.rows));
  taskData->outputs_count.emplace_back(static_cast<uint32_t>(C.columns));
  taskData->outputs_count.emplace_back(static_cast<uint32_t>(C.size()));
  return taskData;
}

TEST(khasanyanov_k_mult_matrix_tests_seq, test_pipeline_run) {
  const int m = 512;
  const int n = 512;

  matrix<double> A = MatrixOperations::generate_random_matrix<double>(m, n, -1000, 1000);
  matrix<double> B = MatrixOperations::generate_random_matrix<double>(m, n, -1000, 1000);
  matrix<double> C{m, n};
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData = create_task_data(A, B, C);
  auto test = std::make_shared<MatrixMultiplication<double>>(taskData);
  auto t = *test;
  EXPECT_TRUE(t.validation());
  t.pre_processing();
  t.run();
  t.post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 8;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(khasanyanov_k_mult_matrix_tests_seq, test_task_run) {
  const int m = 512;
  const int n = 512;

  matrix<double> A = MatrixOperations::generate_random_matrix<double>(m, n, -1000, 1000);
  matrix<double> B = MatrixOperations::generate_random_matrix<double>(m, n, -1000, 1000);
  matrix<double> C{m, n};
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData = create_task_data(A, B, C);
  auto test = std::make_shared<MatrixMultiplication<double>>(taskData);
  auto t = *test;
  EXPECT_TRUE(t.validation());
  t.pre_processing();
  t.run();
  t.post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 8;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}