#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kozlova_e_jacobi_method/include/ops_seq.hpp"

TEST(kozlova_e_jacobi_method, test_pipeline_run) {
  const int N = 500;

  std::vector<double> A(N * N, 1.0);
  for (int i = 0; i < N; ++i) {
    A[i * N + i] = 510.0;
  }
  std::vector<double> B(N, 510.0);
  std::vector<double> X(N, 0.0);
  std::vector<double> expected_X(N, 1.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskDataSeq->inputs_count.emplace_back(N);
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->inputs_count.emplace_back(X.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskDataSeq->outputs_count.emplace_back(X.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kozlova_e_jacobi_method::MethodJacobi>(taskDataSeq);

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
  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(X[i], expected_X[i], 0.5);
  }
}

TEST(kozlova_e_jacobi_method, test_task_run) {
  const int N = 500;

  std::vector<double> A(N * N, 1.0);
  for (int i = 0; i < N; ++i) {
    A[i * N + i] = 510.0;
  }
  std::vector<double> B(N, 510.0);
  std::vector<double> X(N, 0.0);
  std::vector<double> expected_X(N, 1.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskDataSeq->inputs_count.emplace_back(N);
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->inputs_count.emplace_back(X.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskDataSeq->outputs_count.emplace_back(X.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kozlova_e_jacobi_method::MethodJacobi>(taskDataSeq);

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
  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(X[i], expected_X[i], 0.5);
  }
}