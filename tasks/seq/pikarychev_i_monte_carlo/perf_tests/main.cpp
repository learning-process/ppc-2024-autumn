// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/pikarychev_i_monte_carlo/include/ops_seq.hpp"

TEST(sequential_pikarychev_i_monte_carlo_perf_test, test_pipeline_run) {
  const double a = 0.0;
  const double b = 100.0;
  const int num_samples = 1000;
  const int seed = 42;
  double expected_result = 0.0;
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(a, b);
  for (int i = 0; i < num_samples; i++) {
    double x = distribution(generator);
    expected_result += pikarychev_i_monte_carlo_seq::function_double(x);
  }
  expected_result *= (b - a) / num_samples;
  std::vector<double> in = {a, b, static_cast<double>(num_samples), static_cast<double>(seed)};
  std::vector<double> out(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);
  auto testTaskSequential = std::make_shared<pikarychev_i_monte_carlo_seq::TestTaskSequential>(taskDataSeq);
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
  ASSERT_NEAR(expected_result, out[0], 1.0);
}

TEST(sequential_pikarychev_i_monte_carlo_perf_test, test_task_3run) {
  const double a = 0.0;
  const double b = 10.0;
  const double num_samples = 10000.0;
  const int seed = 42;
  double expected_result = 0.0;
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(a, b);
  for (int i = 0; i < num_samples; i++) {
    double x = distribution(generator);
    expected_result += pikarychev_i_monte_carlo_seq::function_double(x);
  }
  expected_result *= (b - a) / num_samples;
  std::vector<double> in = {a, b, static_cast<double>(num_samples), static_cast<double>(seed)};
  std::vector<double> out(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  auto testTaskSequential = std::make_shared<pikarychev_i_monte_carlo_seq::TestTaskSequential>(taskDataSeq);
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
  ASSERT_NEAR(expected_result, out[0], 1.0);
}
