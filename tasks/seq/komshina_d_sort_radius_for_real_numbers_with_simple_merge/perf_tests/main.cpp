#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge, test_pipeline_run) {
  std::vector<double> in = {0.5, 0.2, 0.9, 0.1, 0.4, 0.8, 0.6, 0.3, 0.7, 0.0};
  std::vector<double> out(in.size());

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential>(taskDataSeq);

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

  std::vector<double> sorted_in = in;
  std::sort(sorted_in.begin(), sorted_in.end());

  ASSERT_TRUE(std::equal(out.begin(), out.end(), sorted_in.begin()));
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge, test_task_run) {
  std::vector<double> in = {0.5, 0.2, 0.9, 0.1, 0.4, 0.8, 0.6, 0.3, 0.7, 0.0};
  std::vector<double> out(in.size());

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential>(taskDataSeq);

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

  std::vector<double> sorted_in = in;
  std::sort(sorted_in.begin(), sorted_in.end());

  ASSERT_TRUE(std::equal(out.begin(), out.end(), sorted_in.begin()));
}