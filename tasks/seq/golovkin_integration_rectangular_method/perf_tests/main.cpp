// Golovkin Maksim
#include <gtest/gtest.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/golovkin_integration_rectangular_method/include/ops_seq.hpp"

using namespace golovkin_integration_rectangular_method;
using ppc::core::Perf;
using ppc::core::TaskData;

TEST(golovkin_integration_rectangular_method, test_pipeline_run) {
  const double a = 0.0;
  const double b = 1.0;
  const double epsilon = 0.01;
  const double expected_result = 1.0 / 3.0;
  std::vector<double> in = {a, b, epsilon};
  std::vector<double> out(1, 0.0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  for (const auto& value : in) {
    auto value_ptr = std::make_shared<double>(value);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(value_ptr.get()));
  }
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto integralCalculatorTask = std::make_shared<IntegralCalculator>(taskDataSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();

  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(integralCalculatorTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}
TEST(golovkin_integration_rectangular_method, test_task_run) {
  const double a = 0.0;
  const double b = 1.0;
  const double epsilon = 0.01;
  const double expected_result = 1.0 / 3.0;

  std::vector<double> in = {a, b, epsilon};
  std::vector<double> out(1, 0.0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  for (const auto& value : in) {
    auto* value_ptr = new double(value);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(value_ptr));
  }

  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto integralCalculatorTask = std::make_shared<IntegralCalculator>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(integralCalculatorTask);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}