// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kalyakina_a_trapezoidal_integration_seq/include/ops_seq.hpp"

double function(std::vector<double> input) { return (-3 * pow(input[1], 2) * sin(5 * input[0])) / 2; };

TEST(kalyakina_a_trapezoidal_integration_seq, TrapezoidalIntegrationSequential) {
  // Create data
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {4.0, 6.0}};
  std::vector<unsigned int> intervals = {500, 500};
  std::vector<unsigned int> count(1, 2);
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(count.data()));
  taskDataSequential->inputs_count.emplace_back(count.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
  taskDataSequential->inputs_count.emplace_back(limits.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
  taskDataSequential->inputs_count.emplace_back(intervals.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSequential->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kalyakina_a_trapezoidal_integration_seq::TrapezoidalIntegrationTask>(
      taskDataSequential, function);

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
  ASSERT_NEAR((pow(6.0, 3) - pow(4.0, 3)) * (cos(5 * 1.0) - cos(5 * 0.0)) / 10, out[0], 0.001);
}

TEST(kalyakina_a_trapezoidal_integration_seq, TrapezoidalIntegrationSequential_task_run) {
  // Create data
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {4.0, 6.0}};
  std::vector<unsigned int> intervals = {500, 500};
  std::vector<unsigned int> count(1, 2);
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(count.data()));
  taskDataSequential->inputs_count.emplace_back(count.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
  taskDataSequential->inputs_count.emplace_back(limits.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
  taskDataSequential->inputs_count.emplace_back(intervals.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSequential->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kalyakina_a_trapezoidal_integration_seq::TrapezoidalIntegrationTask>(
      taskDataSequential, function);

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
  ASSERT_NEAR((pow(6.0, 3) - pow(4.0, 3)) * (cos(5 * 1.0) - cos(5 * 0.0)) / 10, out[0], 0.001);
}
