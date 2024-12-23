// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/drozhdinov_d_mult_matrix_fox/include/ops_seq.hpp"
using namespace drozhdinov_d_mult_matrix_fox_seq;
namespace drozhdinov_d_mult_matrix_fox_seq {
std::vector<double> MatrixMult(const std::vector<double> &A, const std::vector<double> &B, int k, int l, int n) {
  std::vector<double> result(k * n, 0.0);

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      for (int p = 0; p < l; p++) {
        result[i * n + j] += A[i * l + p] * B[p * n + j];
      }
    }
  }

  return result;
}

std::vector<double> getRandomMatrix(int sz, int lbound, int rbound) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> vec(sz);
  std::uniform_int_distribution<int> dist(lbound, rbound);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace drozhdinov_d_mult_matrix_fox_seq
TEST(drozhdinov_d_mult_matrix_fox_seq_perf_test, test_pipeline_run) {
  int k = 250;
  int l = 250;
  int m = 250;
  int n = 250;
  std::vector<double> A = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(m * n, -100, 100);
  std::vector<double> res(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->inputs_count.emplace_back(l);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(k);
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  auto testTaskSequential = std::make_shared<drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 50;
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
  for (int i = 0; i < k * n; i++) {
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
  }
}

TEST(drozhdinov_d_mult_matrix_fox_seq_perf_test, test_task_run) {
  int k = 250;
  int l = 250;
  int m = 250;
  int n = 250;
  std::vector<double> A = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(m * n, -100, 100);
  std::vector<double> res(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->inputs_count.emplace_back(l);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(k);
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  auto testTaskSequential = std::make_shared<drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 50;
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
  for (int i = 0; i < k * n; i++) {
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
  }
}