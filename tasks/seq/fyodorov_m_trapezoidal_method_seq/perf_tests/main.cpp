// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/fyodorov_m_trapezoidal_method_seq/include/ops_seq.hpp"

namespace {

bool almost_equal(double a, double b, double epsilon = 1e-6) { return std::abs(a - b) < epsilon; }

double test_func_1(const std::vector<double>& x) { return x[0]; }

}  // namespace

TEST(sequential_example_perf_test, test_int_task_pipeline_run) {
  // Create data
  std::function<double(const std::vector<double>&)> func = test_func_1;
  std::vector<double> lower_bounds = {0.0};
  std::vector<double> upper_bounds = {1.0};
  std::vector<int> intervals = {100};
  std::vector<double> out(1, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
  taskDataSeq->inputs_count.emplace_back(lower_bounds.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
  taskDataSeq->inputs_count.emplace_back(upper_bounds.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
  taskDataSeq->inputs_count.emplace_back(intervals.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  auto testTaskSequential = std::make_shared<fyodorov_m_trapezoidal_method_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_TRUE(almost_equal(out[0], 0.5));
}

TEST(sequential_example_perf_test, test_int_task_task_run) {
  // Create data
  std::function<double(const std::vector<double>&)> func = test_func_1;
  std::vector<double> lower_bounds = {0.0};
  std::vector<double> upper_bounds = {1.0};
  std::vector<int> intervals = {100};
  std::vector<double> out(1, 0.0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&func));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_bounds.data()));
  taskDataSeq->inputs_count.emplace_back(lower_bounds.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_bounds.data()));
  taskDataSeq->inputs_count.emplace_back(upper_bounds.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
  taskDataSeq->inputs_count.emplace_back(intervals.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<fyodorov_m_trapezoidal_method_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_TRUE(almost_equal(out[0], 0.5));
}