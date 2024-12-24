#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_seq.hpp"

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq {

std::vector<double> getRandomMatrix(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-1000, 1000);
  std::vector<double> matrix(sz);
  for (int i = 0; i < sz; ++i) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

double Ax_b(int n, int m, std::vector<double> a, std::vector<double> res) {
  std::vector<double> tmp(m, 0);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n - 1; ++j) {
      tmp[i] += a[i * n + j] * res[j];
    }
    tmp[i] -= a[i * n + m];
  }

  double tmp_norm = 0;
  for (int i = 0; i < m; i++) {
    tmp_norm += tmp[i] * tmp[i];
  }
  return sqrt(tmp_norm);
}

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_pipeline_run) {
  const int cols = 101;
  const int rows = 100;

  std::vector<double> matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::getRandomMatrix(cols * rows);
  std::vector<double> res(cols - 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  auto MPIGaussHorizontalSequential =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential>(
          taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto time_start = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - time_start).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIGaussHorizontalSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_NEAR(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::Ax_b(cols, rows, matrix, res), 0, 1e-6);
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_task_run) {
  const int cols = 101;
  const int rows = 100;

  std::vector<double> matrix = shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::getRandomMatrix(cols * rows);
  std::vector<double> res(cols - 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  auto MPIGaussHorizontalSequential =
      std::make_shared<shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential>(
          taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto time_start = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - time_start).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIGaussHorizontalSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_NEAR(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::Ax_b(cols, rows, matrix, res), 0, 1e-6);
}
