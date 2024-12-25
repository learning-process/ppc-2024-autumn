#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_pipeline_run) {
  std::vector<double> in = {3.0, -1.0, 7.5, 0.0, -8.2, 15.5, -3.4, 0.1, 9.8, 4.0};
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
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

  std::vector<double> expected = in;
  std::sort(expected.begin(), expected.end());
  for (size_t i = 0; i < in.size(); ++i) {
    ASSERT_EQ(out[i], expected[i]) << "Failed at index " << i;
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq, test_task_run) {
  std::vector<double> in = {3.0, -1.0, 7.5, 0.0, -8.2, 15.5, -3.4, 0.1, 9.8, 4.0};
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
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

   std::vector<double> expected = in;
  std::sort(expected.begin(), expected.end());
  for (size_t i = 0; i < in.size(); ++i) {
    ASSERT_EQ(out[i], expected[i]) << "Failed at index " << i;
  }
}