#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/bessonov_e_integration_monte_carlo/include/ops_seq.hpp"

TEST(bessonov_e_integration_monte_carlo_seq, TestPipelineRun) {
  double a = 0.0;
  double b = 2.0;
  int num_points = 10000000;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&num_points));
  double output = 1.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  auto testTaskSequential = std::make_shared<bessonov_e_integration_monte_carlo_seq::TestTaskSequential>(taskData);
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
  double expected_result = 4.0;
  ASSERT_NEAR(output, expected_result, 1e-1);
}

TEST(bessonov_e_integration_monte_carlo_seq, TestTaskRun) {
  double a = 0.0;
  double b = 2.0;
  int num_points = 10000000;
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&a));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&b));
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(&num_points));
  double output = 1.0;
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(&output));
  auto testTaskSequential = std::make_shared<bessonov_e_integration_monte_carlo_seq::TestTaskSequential>(taskData);
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
  double expected_result = 4.0;
  ASSERT_NEAR(output, expected_result, 1e-1);
}
