#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/rams_s_radix_sort_with_simple_merge_for_doubles/include/ops_seq.hpp"

void rams_s_radix_sort_with_simple_merge_for_doubles_seq_run_perf_test(bool pipeline) {
  size_t length = 500000;
  std::vector<double> in(length, 0);
  std::random_device dev;
  std::mt19937_64 gen(dev());
  for (size_t i = 0; i < length; i++) {
    while (std::isnan(in[i] = std::bit_cast<double>(gen())));
  }
  std::vector<double> out(length, 0);
  std::vector<double> expected(in);
  std::sort(expected.begin(), expected.end());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<rams_s_radix_sort_with_simple_merge_for_doubles_seq::TaskSequential>(taskDataSeq);

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
  if (pipeline) {
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
  } else {
    perfAnalyzer->task_run(perfAttr, perfResults);
  }
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expected, out);
}

TEST(rams_s_radix_sort_with_simple_merge_for_doubles_seq_perf_test, test_pipeline_run) {
  rams_s_radix_sort_with_simple_merge_for_doubles_seq_run_perf_test(true);
}

TEST(rams_s_radix_sort_with_simple_merge_for_doubles_seq_perf_test, test_task_run) {
  rams_s_radix_sort_with_simple_merge_for_doubles_seq_run_perf_test(false);
}
