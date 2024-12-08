#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/khokhlov_a_iterative_seidel_method/include/ops_seq_khokhlov.hpp"

TEST(khokhlov_a_iterative_seidel_method_seq, test_pipline_run_seq) {
  const int n = 800;
  const int maxiter = 800;
  const double eps = 1e-3;

  // create data
  std::vector<double> A(n * n, 0.0);
  std::vector<double> b(n, 0.0);
  std::vector<double> result(n, 0.0);
  khokhlov_a_iterative_seidel_method_seq::getRandomSLAU(A, b, n);

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskdataSeq->inputs_count.emplace_back(n);
  taskdataSeq->inputs_count.emplace_back(maxiter);
  taskdataSeq->inputs_count.emplace_back(eps);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskdataSeq->outputs_count.emplace_back(result.size());

  // crate task
  auto testTaskSeq = std::make_shared<khokhlov_a_iterative_seidel_method_seq::seidel_method_seq>(taskdataSeq);

  // create perf attrib
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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(b.size(), result.size());
}

TEST(khokhlov_a_iterative_seidel_method_seq, test_task_run_seq) {
  const int n = 800;
  const int maxiter = 800;
  const double eps = 1e-3;

  // create data
  std::vector<double> A(n * n, 0.0);
  std::vector<double> b(n, 0.0);
  std::vector<double> result(n, 0.0);
  khokhlov_a_iterative_seidel_method_seq::getRandomSLAU(A, b, n);

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskdataSeq->inputs_count.emplace_back(n);
  taskdataSeq->inputs_count.emplace_back(maxiter);
  taskdataSeq->inputs_count.emplace_back(eps);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskdataSeq->outputs_count.emplace_back(result.size());

  // crate task
  auto testTaskSeq = std::make_shared<khokhlov_a_iterative_seidel_method_seq::seidel_method_seq>(taskdataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(b.size(), result.size());
}