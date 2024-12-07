#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/sozonov_i_gaussian_method_horizontal_strip_scheme/include/ops_seq.hpp"

namespace sozonov_i_gaussian_method_horizontal_strip_scheme_seq {

std::vector<double> getRandomMatrix(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dis(-1000, 1000);
  std::vector<double> mat(sz);
  for (int i = 0; i < sz; ++i) {
    mat[i] = dis(gen);
  }
  return mat;
}

double Ax_b(int n, int m, std::vector<double> a, std::vector<double> x) {
  std::vector<double> c(m, 0);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n - 1; ++j) {
      c[i] += a[i * n + j] * x[j];
    }
    c[i] -= a[i * n + m];
  }

  double c_norm = 0;
  for (int i = 0; i < m; i++) {
    c_norm += c[i] * c[i];
  }
  return sqrt(c_norm);
}

}  // namespace sozonov_i_gaussian_method_horizontal_strip_scheme_seq

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_pipeline_run) {
  const double EPS = 1e-6;

  const int cols = 101;
  const int rows = 100;

  // Create data
  std::vector<double> in = sozonov_i_gaussian_method_horizontal_strip_scheme_seq::getRandomMatrix(cols * rows);
  std::vector<double> out(cols - 1, 0);
  double ans;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential>(taskDataSeq);

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

  ans = sozonov_i_gaussian_method_horizontal_strip_scheme_seq::Ax_b(cols, rows, in, out);
  ASSERT_NEAR(ans, 0, EPS);
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_seq, test_task_run) {
  const double EPS = 1e-6;

  const int cols = 101;
  const int rows = 100;

  // Create data
  std::vector<double> in = sozonov_i_gaussian_method_horizontal_strip_scheme_seq::getRandomMatrix(cols * rows);
  std::vector<double> out(cols - 1, 0);
  double ans;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<sozonov_i_gaussian_method_horizontal_strip_scheme_seq::TestTaskSequential>(taskDataSeq);

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

  ans = sozonov_i_gaussian_method_horizontal_strip_scheme_seq::Ax_b(cols, rows, in, out);
  ASSERT_NEAR(ans, 0, EPS);
}
