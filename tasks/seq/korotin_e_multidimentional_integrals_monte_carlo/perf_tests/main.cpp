// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/korotin_e_multidimentional_integrals_monte_carlo/include/ops_seq.hpp"

namespace korotin_e_multidimentional_integrals_monte_carlo_seq {

double test_func(double *x, int x_size) { return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]; }

}  // namespace korotin_e_multidimentional_integrals_monte_carlo_seq

TEST(korotin_e_multidimentional_integrals_monte_carlo_seq, test_pipeline_run) {
  std::vector<std::pair<double, double>> borders(3);
  std::vector<double> res(1, 0);
  std::vector<size_t> N(1, 500);
  std::vector<double (*)(double *, int)> F(1, &korotin_e_multidimentional_integrals_monte_carlo_seq::test_func);
  double ref = 32.0;

  borders[0] = borders[1] = borders[2] = std::pair<double, double>(0.0, 2.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(F.data()));
  taskDataSeq->inputs_count.emplace_back(F.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(borders.data()));
  taskDataSeq->inputs_count.emplace_back(borders.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(N.data()));
  taskDataSeq->inputs_count.emplace_back(N.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<korotin_e_multidimentional_integrals_monte_carlo_seq::TestTaskSequential>(taskDataSeq);

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

  double err = testTaskSequential->possible_error();
  bool ans = (std::abs(res[0] - ref) < err);

  ASSERT_EQ(ans, true);
}

TEST(korotin_e_multidimentional_integrals_monte_carlo_seq, test_task_run) {
  std::vector<std::pair<double, double>> borders(3);
  std::vector<double> res(1, 0);
  std::vector<size_t> N(1, 500);
  std::vector<double (*)(double *, int)> F(1, &korotin_e_multidimentional_integrals_monte_carlo_seq::test_func);

  double ref = 32.0;

  borders[0] = borders[1] = borders[2] = std::pair<double, double>(0.0, 2.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(F.data()));
  taskDataSeq->inputs_count.emplace_back(F.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(borders.data()));
  taskDataSeq->inputs_count.emplace_back(borders.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(N.data()));
  taskDataSeq->inputs_count.emplace_back(N.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<korotin_e_multidimentional_integrals_monte_carlo_seq::TestTaskSequential>(taskDataSeq);

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

  double err = testTaskSequential->possible_error();
  bool ans = (std::abs(res[0] - ref) < err);

  ASSERT_EQ(ans, true);
}
