// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/veliev_e_sobel_operator/include/ops_seq.hpp"

namespace veliev_e_sobel_operator_seq {
std::vector<double> create_random_vector(int size) {
  std::uniform_real_distribution<double> unif(static_cast<double>(0), static_cast<double>(255));
  std::random_device rand_dev;
  std::mt19937 rand_engine(rand_dev());
  std::vector<double> tmp;
  tmp.reserve(size);
  std::generate_n(std::back_inserter(tmp), size, [&]() { return unif(rand_engine); });

  return tmp;
}
}  // namespace veliev_e_sobel_operator_seq

TEST(veliev_e_sobel_operator_seq, pipeline) {
  // Create data
  int h = 50000;
  int w = 1000;
  std::vector<double> in = veliev_e_sobel_operator_seq::create_random_vector(50000000);
  std::vector<double> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<veliev_e_sobel_operator_seq::TestTaskSequential>(taskDataSeq);

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
TEST(veliev_e_sobel_operator_seq, task) {
  // Create data
  int h = 50000;
  int w = 1000;
  std::vector<double> in = veliev_e_sobel_operator_seq::create_random_vector(50000000);
  std::vector<double> out(in.size(), 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs_count.emplace_back(h);
  taskDataSeq->inputs_count.emplace_back(w);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<veliev_e_sobel_operator_seq::TestTaskSequential>(taskDataSeq);

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
