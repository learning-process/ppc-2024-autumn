#include <memory>

#include "../include/matrix_operations.hpp"
#include "../include/tests.hpp"
#include "core/perf/include/perf.hpp"

using namespace khasanyanov_k_fox_algorithm;

TEST(khasanyanov_k_mult_matrix_tests_seq, test_pipeline_run) {
  const int m = 512;
  const int n = 512;

  matrix<double> A = MatrixOperations::generate_random_matrix<double>(m, n, -1000, 1000);
  matrix<double> B = MatrixOperations::generate_random_matrix<double>(m, n, -1000, 1000);
  matrix<double> C{m, n};
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData = create_task_data(A, B, C);
  auto test = std::make_shared<MatrixMultiplication<double>>(taskData);
  RUN_TASK(*test);
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
  RUN_TASK(*test);
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