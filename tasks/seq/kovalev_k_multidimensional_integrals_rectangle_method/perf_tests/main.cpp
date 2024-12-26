#include <gtest/gtest.h>

#include <numbers>

#include "core/perf/include/perf.hpp"
#include "seq/kovalev_k_multidimensional_integrals_rectangle_method/include/header.hpp"

namespace kovalev_k_multidimensional_integrals_rectangle_method_seq {
double f1Euler(std::vector<double> &arguments) { return 2 * std::cos(arguments.at(0)) * std::sin(arguments.at(0)); }
double f3advanced(std::vector<double> &arguments) {
  return std::sin(arguments.at(0)) * std::tan(arguments.at(1)) * std::log(arguments.at(2));
}
}  // namespace kovalev_k_multidimensional_integrals_rectangle_method_seq

TEST(kovalev_k_multidimensional_integrals_rectangle_method_seq, test_pipeline_run) {
  const size_t dim = 1;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = 0;
  lims[0].second = 0.5 * std::numbers::pi;
  double h = 0.0005;
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(lims.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  auto testTaskSequential = std::make_shared<
      kovalev_k_multidimensional_integrals_rectangle_method_seq::MultidimensionalIntegralsRectangleMethod>(
      taskSeq, kovalev_k_multidimensional_integrals_rectangle_method_seq::f1Euler);
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
  ASSERT_NEAR(1.0, out[0], eps);
}

TEST(kovalev_k_multidimensional_integrals_rectangle_method_seq, test_task_run) {
  const size_t dim = 3;
  std::vector<std::pair<double, double>> lims(dim);
  lims[0].first = 0.8;
  lims[0].second = 1.0;
  lims[1].first = 1.9;
  lims[1].second = 2.0;
  lims[2].first = 2.9;
  lims[2].second = 3.0;
  double h = 0.005;
  double eps = 1e-3;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(lims.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(lims.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&h));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  auto testTaskSequential = std::make_shared<
      kovalev_k_multidimensional_integrals_rectangle_method_seq::MultidimensionalIntegralsRectangleMethod>(
      taskSeq, kovalev_k_multidimensional_integrals_rectangle_method_seq::f3advanced);
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
  ASSERT_NEAR(-0.00427191467841401, out[0], eps);
}