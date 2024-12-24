#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/bessonov_e_multi_integration_trapezoid_method/include/ops_seq.hpp"

TEST(bessonov_e_multi_integration_trapezoid_method_seq, TestPipelineRun) {
  size_t dim = 2;
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {std::numbers::pi / 2, std::numbers::pi / 2};
  std::vector<int> steps = {1000, 1000};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  auto testTaskSequential =
      std::make_shared<bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential>(taskData);
  testTaskSequential->integrand = [](const std::vector<double>& point) { return cos(point[0]) * sin(point[1]); };

  ASSERT_TRUE(testTaskSequential->validation());
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

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

  double expected_result = 1.0;
  ASSERT_NEAR(result, expected_result, 1e-2);
}

TEST(bessonov_e_multi_integration_trapezoid_method_seq, TestTaskRun) {
  size_t dim = 2;
  std::vector<double> lower_limits = {0.0, 0.0};
  std::vector<double> upper_limits = {std::numbers::pi / 2, std::numbers::pi / 2};
  std::vector<int> steps = {1000, 1000};
  double result = 0.0;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dim));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower_limits.data()));
  taskData->inputs_count.emplace_back(lower_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper_limits.data()));
  taskData->inputs_count.emplace_back(upper_limits.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  taskData->inputs_count.emplace_back(steps.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  auto testTaskSequential =
      std::make_shared<bessonov_e_multi_integration_trapezoid_method_seq::TestTaskSequential>(taskData);
  testTaskSequential->integrand = [](const std::vector<double>& point) { return cos(point[0]) * sin(point[1]); };

  ASSERT_TRUE(testTaskSequential->validation());
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

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

  double expected_result = 1.0;
  ASSERT_NEAR(result, expected_result, 1e-2);
}