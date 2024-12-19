#include <gtest/gtest.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cmath>

#include "core/perf/include/perf.hpp"
#include "seq/shulpin_monte_carlo_integration/include/monte_carlo_integral.hpp"

constexpr double ESTIMATE = 1e-3;

TEST(shulpin_monte_carlo_integration, test_pipeline_run) {
  double a = 0.0;
  double b = M_PI;
  int N = 1000000;

  const double expected_result = 2.0;

  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  taskDataSeq->outputs_count.emplace_back(1);

  auto testTaskSequential = std::make_shared<shulpin_monte_carlo_integration::TestMPITaskSequential>(taskDataSeq);

  testTaskSequential->set_seq(shulpin_monte_carlo_integration::fsin);

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

  ASSERT_LT(std::abs(output - expected_result), ESTIMATE);
}

TEST(shulpin_monte_carlo_integration, test_task_run) {
  double a = 0.0;
  double b = M_PI;
  int N = 1000000;
  const double expected_result = 2.0;

  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  taskDataSeq->outputs_count.emplace_back(1);

  auto testTaskSequential = std::make_shared<shulpin_monte_carlo_integration::TestMPITaskSequential>(taskDataSeq);

  testTaskSequential->set_seq(shulpin_monte_carlo_integration::fsin);

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

  ASSERT_LT(std::abs(output - expected_result), ESTIMATE);
}