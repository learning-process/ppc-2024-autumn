// Copyright 2024 Lupsha Egor
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/lupsha_e_rect_integration/include/ops_seq.hpp"

std::tuple<double, double, int> generate_random_data() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> bounds_dist(0.0, 10.0);
  std::uniform_int_distribution<> intervals_dist(100000, 2000000);

  double lower_bound = bounds_dist(gen);
  double upper_bound = lower_bound + bounds_dist(gen);
  int num_intervals = intervals_dist(gen);

  return std::make_tuple(lower_bound, upper_bound, num_intervals);
}

TEST(lupsha_e_rect_integration_seq, test_pipeline_run) {
  double lower_bound = 0.0;
  double upper_bound = 2.0;
  int num_intervals = 1000000;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_intervals));
  taskDataSeq->inputs_count.emplace_back(3);

  std::vector<double> result(1, 0.0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  auto TestTaskSequential = std::make_shared<lupsha_e_rect_integration_seq::TestTaskSequential>(taskDataSeq);
  TestTaskSequential->function_set([](double x) { return x * x; });

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  EXPECT_NEAR(result[0], (8.0 / 3.0), 0.0001);
}

TEST(lupsha_e_rect_integration_seq, test_task_run) {
  double lower_bound = 0.0;
  double upper_bound = 2.0;
  int num_intervals = 1000000;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_intervals));
  taskDataSeq->inputs_count.emplace_back(3);

  std::vector<double> result(1, 0.0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  auto TestTaskSequential = std::make_shared<lupsha_e_rect_integration_seq::TestTaskSequential>(taskDataSeq);
  TestTaskSequential->function_set([](double x) { return x * x; });

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  EXPECT_NEAR(result[0], (8.0 / 3.0), 0.0001);
}

TEST(lupsha_e_rect_integration_seq, test_pipeline_run_random) {
  auto [lower_bound, upper_bound, num_intervals] = generate_random_data();

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&lower_bound));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&upper_bound));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_intervals));
  taskDataSeq->inputs_count.emplace_back(3);

  std::vector<double> result(1, 0.0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  auto TestTaskSequential = std::make_shared<lupsha_e_rect_integration_seq::TestTaskSequential>(taskDataSeq);
  TestTaskSequential->function_set([](double x) { return x * x; });

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;

  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  double expected_result = (std::pow(upper_bound, 3) - std::pow(lower_bound, 3)) / 3.0;
  EXPECT_NEAR(result[0], expected_result, 0.0001);
}