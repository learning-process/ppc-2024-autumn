#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/prokhorov_n_rectangular_integration/include/ops_seq.hpp"

TEST(prokhorov_n_rectangular_integration, test_perf_integration_x_squared) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n = 100000;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return x * x; });

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

  const double expected_result = 1.0 / 3.0;
  ASSERT_NEAR(out[0], expected_result, 1e-3);
}
TEST(prokhorov_n_rectangular_integration, test_perf_integration_sin_x) {
  const double lower_bound = 0.0;
  const double upper_bound = M_PI;
  const int n = 100000;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return std::sin(x); });

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

  const double expected_result = 2.0;
  ASSERT_NEAR(out[0], expected_result, 1e-3);
}
TEST(prokhorov_n_rectangular_integration, test_perf_integration_exp_x) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n = 100000;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return std::exp(x); });

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

  const double expected_result = std::exp(1.0) - 1.0;
  ASSERT_NEAR(out[0], expected_result, 1e-3);
}
TEST(prokhorov_n_rectangular_integration, test_perf_integration_one_over_x) {
  const double lower_bound = 1.0;
  const double upper_bound = 10.0;
  const int n = 100000;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return 1.0 / x; });

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

  const double expected_result = std::log(10.0);
  ASSERT_NEAR(out[0], expected_result, 1e-3);
}
