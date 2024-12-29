#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/perf/include/perf.hpp"

static std::vector<double> randv(size_t sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> vec(sz);
  for (size_t i = 0; i < sz; i++) {
    vec[i] = -50 + (gen() / 10.);
  }
  return vec;
}

TEST(sorochkin_d_radix_sort_batcher_merge_double_seq_perf_test, test_pipeline_run) {
  // Create data
  std::vector<double> in = randv(128'000);
  std::vector<double> out(in.size(), 0.);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  // in
  taskDataSeq->inputs = {reinterpret_cast<uint8_t *>(in.data())};
  taskDataSeq->inputs_count = {static_cast<uint32_t>(in.size())};
  // out
  taskDataSeq->outputs = {reinterpret_cast<uint8_t *>(out.data())};
  taskDataSeq->outputs_count = {static_cast<uint32_t>(out.size())};

  // Create Task
  auto testTaskSequential =
      std::make_shared<sorochkin_d_radix_sort_batcher_merge_double_seq::TestTaskSequential>(taskDataSeq);

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

TEST(sorochkin_d_radix_sort_batcher_merge_double_seq_perf_test, test_task_run) {
  // Create data
  std::vector<double> in = randv(128'000);
  std::vector<double> out(in.size(), 0.);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  // in
  taskDataSeq->inputs = {reinterpret_cast<uint8_t *>(in.data())};
  taskDataSeq->inputs_count = {static_cast<uint32_t>(in.size())};
  // out
  taskDataSeq->outputs = {reinterpret_cast<uint8_t *>(out.data())};
  taskDataSeq->outputs_count = {static_cast<uint32_t>(out.size())};

  // Create Task
  auto testTaskSequential =
      std::make_shared<sorochkin_d_radix_sort_batcher_merge_double_seq::TestTaskSequential>(taskDataSeq);

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
