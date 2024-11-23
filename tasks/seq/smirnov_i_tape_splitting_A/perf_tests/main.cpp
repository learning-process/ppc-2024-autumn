#include <gtest/gtest.h>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/smirnov_i_tape_splitting_A/include/ops_seq.hpp"

TEST(smirnov_i_tape_splitting_A_seq, test_pipeline_run) {
  int m_a = 2;
  int n_a = 3;
  int m_b = 3;
  int n_b = 4;
  double* A = new double[m_a * n_a];
  double* B = new double[m_b * n_b];
  for (int i = 0; i < n_a * m_a; i++) {
    A[i] = i;
  }
  for (int i = 0; i < n_b * m_b; i++) {
    B[i] = 2 * i;
  }
  double expected_res[] = {40, 46, 52, 58, 112, 136, 160, 184};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

  taskDataSeq->inputs_count.emplace_back(m_a);
  taskDataSeq->inputs_count.emplace_back(n_a);
  taskDataSeq->inputs_count.emplace_back(m_b);
  taskDataSeq->inputs_count.emplace_back(n_b);

  double* res = new double[n_a * m_b];
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
  taskDataSeq->outputs_count.emplace_back(n_a);
  taskDataSeq->outputs_count.emplace_back(m_a);

  auto TestTaskSequential = std::make_shared<smirnov_i_tape_splitting_A::TestMPITaskSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 0; i < m_a * n_b; i++) {
    ASSERT_NEAR(expected_res[i], res[i], 1e-6);
  }
  delete[] A;
  delete[] B;
  delete[] res;
}

TEST(smirnov_i_tape_splitting_A_seq, test_task_run) {
  int m_a = 2;
  int n_a = 3;
  int m_b = 3;
  int n_b = 4;
  double* A = new double[m_a * n_a];
  double* B = new double[m_b * n_b];
  for (int i = 0; i < n_a * m_a; i++) {
    A[i] = i;
  }
  for (int i = 0; i < n_b * m_b; i++) {
    B[i] = 2 * i;
  }
  double expected_res[] = {40, 46, 52, 58, 112, 136, 160, 184};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B));

  taskDataSeq->inputs_count.emplace_back(m_a);
  taskDataSeq->inputs_count.emplace_back(n_a);
  taskDataSeq->inputs_count.emplace_back(m_b);
  taskDataSeq->inputs_count.emplace_back(n_b);

  double* res = new double[n_a * m_b];
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res));
  taskDataSeq->outputs_count.emplace_back(n_a);
  taskDataSeq->outputs_count.emplace_back(m_a);

  auto TestTaskSequential = std::make_shared<smirnov_i_tape_splitting_A::TestMPITaskSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  for (int i = 0; i < m_a * n_b; i++) {
    ASSERT_NEAR(expected_res[i], res[i], 1e-6);
  }
  delete[] A;
  delete[] B;
  delete[] res;
}
