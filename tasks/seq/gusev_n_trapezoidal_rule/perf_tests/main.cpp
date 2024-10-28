// Copyright 2024 Nesterov Alexander
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "core/perf/include/perf.hpp"
#include "seq/gusev_n_trapezoidal_rule/include/ops_seq.hpp"
#include <corecrt_math_defines.h>

TEST(gusev_n_trapezoidal_rule_seq, test_pipeline_run) {
  const double a = 0.0;
  const double b = 1.0;
  const int n = 10000000;

  const double expected_result = 1.0 / 3.0;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<gusev_n_trapezoidal_rule_seq::TrapezoidalIntegrationSequential>(taskDataSeq);

  std::function<double(double)> func = [](double x) { return x * x; };
  testTaskSequential->set_function(func);

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

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(gusev_n_trapezoidal_rule_seq, test_task_run) {
  const double a = 0.0;
  const double b = 1.0;
  const int n = 10000000;
  const double expected_result = 1.0 / 3.0;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<gusev_n_trapezoidal_rule_seq::TrapezoidalIntegrationSequential>(taskDataSeq);

  std::function<double(double)> func = [](double x) { return x * x; };
  testTaskSequential->set_function(func);

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

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}
