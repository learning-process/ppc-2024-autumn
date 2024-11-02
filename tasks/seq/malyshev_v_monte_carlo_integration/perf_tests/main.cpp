#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/malyshev_v_monte_carlo_integration/include/malyshev_v_monte_carlo_integration.hpp"

using namespace malyshev_v_monte_carlo_integration;

TEST(malyshev_v_monte_carlo_integration, test_pipeline_run) {
  double a = 0.0;
  double b = 1.0;
  int num_points = 1000000;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&num_points));
  double output = 1.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&output));

  auto monteCarloTask = std::make_shared<MonteCarloIntegration>(taskData);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(monteCarloTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  double expected_result = 1.0 / 3.0;  
  ASSERT_NEAR(output, expected_result, 0.1);
}

TEST(malyshev_v_monte_carlo_integration, test_task_run) {
  double a = -1.0;
  double b = 1.0;
  int num_points = 1000000;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(&num_points));
  double output = 1.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(&output));

  auto monteCarloTask = std::make_shared<MonteCarloIntegration>(taskData);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(monteCarloTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);

  double expected_result = 2.0 / 3.0;  
  ASSERT_NEAR(output, expected_result, 0.1);
}
