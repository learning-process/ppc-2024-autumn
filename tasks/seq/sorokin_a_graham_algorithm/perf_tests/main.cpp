// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numbers>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/sorokin_a_graham_algorithm/include/ops_seq.hpp"

namespace sorokin_a_graham_algorithm_seq {
std::vector<int> getrndvec(int n, int radius) {
  if (n % 2 != 0) {
    throw std::invalid_argument("The number of elements n must be even.");
  }
  std::random_device rand_dev;
  std::mt19937 rand_engine(rand_dev());
  std::uniform_real_distribution<double> dist_radius(0.0, static_cast<double>(radius));
  std::uniform_real_distribution<double> dist_angle(0.0, 2.0 * std::numbers::pi);
  std::vector<int> tmp(n);
  for (int i = 0; i < n / 2; ++i) {
    double r = dist_radius(rand_engine);
    double theta = dist_angle(rand_engine);
    double x = r * cos(theta);
    double y = r * sin(theta);
    tmp[2 * i] = static_cast<int>(x);
    tmp[2 * i + 1] = static_cast<int>(y);
  }

  return tmp;
}
}  // namespace sorokin_a_graham_algorithm_seq

TEST(sorokin_a_graham_algorithm_seq, Elem1000000) {
  // Create data
  std::vector<int> in = sorokin_a_graham_algorithm_seq::getrndvec(2000000, 100);
  std::vector<int> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<sorokin_a_graham_algorithm_seq::TestTaskSequential>(taskDataSeq);

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
}

TEST(sorokin_a_graham_algorithm_seq, Elem1000000task) {
  // Create data
  std::vector<int> in = sorokin_a_graham_algorithm_seq::getrndvec(2000000, 100);
  std::vector<int> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<sorokin_a_graham_algorithm_seq::TestTaskSequential>(taskDataSeq);

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
}
