#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shvedova_v_multi_integral_simpson/include/ops_seq.hpp"

namespace shvedova_v_multidimensional_integral_simpson_seq {
double tripleComplexFunction(std::vector<double>& args) {
  double sum_of_squares = 0.0;
  double sum_of_linear = 0.0;
  for (size_t i = 0; i < args.size(); ++i) {
    sum_of_squares += args[i] * args[i];
    sum_of_linear += args[i];
  }
  return std::sin(sum_of_squares) * std::exp(-sum_of_linear);
}
}  // namespace shvedova_v_multidimensional_integral_simpson_seq

TEST(shvedova_v_multidimensional_integral_simpson_seq, test_pipeline_run) {
  std::vector<std::pair<double, double>> integration_limits = {
      {0.0, 10.0},
      {0.0, 9.0},
      {0.0, 8.0},
  };

  shvedova_v_multidimensional_integral_simpson_seq::FunctionType target_function = [](std::vector<double>& args) {
    return shvedova_v_multidimensional_integral_simpson_seq::tripleComplexFunction(args);
  };

  double tolerance = 1e-3;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(integration_limits.data()));
  taskDataSeq->inputs_count.emplace_back(integration_limits.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target_function));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  taskDataSeq->inputs_count.emplace_back(1);

  double result = 0.0;
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  taskDataSeq->outputs_count.emplace_back(1);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(
      std::make_shared<shvedova_v_multidimensional_integral_simpson_seq::MultidimensionalIntegralSequential>(
          taskDataSeq));
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  EXPECT_TRUE(result > 0.0);
}

TEST(shvedova_v_multidimensional_integral_simpson_seq, test_task_run) {
  std::vector<std::pair<double, double>> integration_limits = {
      {0.0, 10.0},
      {0.0, 9.0},
      {0.0, 8.0},
  };

  shvedova_v_multidimensional_integral_simpson_seq::FunctionType target_function = [](std::vector<double>& args) {
    return shvedova_v_multidimensional_integral_simpson_seq::tripleComplexFunction(args);
  };

  double tolerance = 1e-3;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(integration_limits.data()));
  taskDataSeq->inputs_count.emplace_back(integration_limits.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target_function));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  taskDataSeq->inputs_count.emplace_back(1);

  double result = 0.0;
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  taskDataSeq->outputs_count.emplace_back(1);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(
      std::make_shared<shvedova_v_multidimensional_integral_simpson_seq::MultidimensionalIntegralSequential>(
          taskDataSeq));
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  EXPECT_TRUE(result > 0.0);
}