#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/korablev_v_quick_sort_simple_merge/include/ops_seq.hpp"

namespace korablev_v_qucik_sort_simple_merge_seq {
std::vector<double> generate_random_vector(size_t size, double min_val = -1000.0, double max_val = 1000.0) {
  std::vector<double> data(size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min_val, max_val);

  for (auto& val : data) {
    val = dist(gen);
  }

  return data;
}
}  // namespace korablev_v_qucik_sort_simple_merge_seq

TEST(korablev_v_quick_sort_simple_merge_seq, test_pipeline_run) {
  const size_t vector_size = 10000;
  std::vector<size_t> in_size(1, vector_size);
  std::vector<double> input_data = korablev_v_qucik_sort_simple_merge_seq::generate_random_vector(vector_size);
  std::vector<double> output_data(vector_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(output_data.size());

  korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential quickSortTask(taskDataSeq);
  ASSERT_TRUE(quickSortTask.validation());
  quickSortTask.pre_processing();
  quickSortTask.run();
  quickSortTask.post_processing();

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
      std::make_shared<korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential>(taskDataSeq));
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_TRUE(std::is_sorted(output_data.begin(), output_data.end()));
}

TEST(korablev_v_quick_sort_simple_merge_seq, test_task_run) {
  const size_t vector_size = 10000;
  std::vector<size_t> in_size(1, vector_size);
  std::vector<double> input_data = korablev_v_qucik_sort_simple_merge_seq::generate_random_vector(vector_size);
  std::vector<double> output_data(vector_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataSeq->inputs_count.emplace_back(in_size.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(output_data.size());

  korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential quickSortTask(taskDataSeq);
  ASSERT_TRUE(quickSortTask.validation());
  quickSortTask.pre_processing();
  quickSortTask.run();
  quickSortTask.post_processing();

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
      std::make_shared<korablev_v_qucik_sort_simple_merge_seq::QuickSortSimpleMergeSequential>(taskDataSeq));
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_TRUE(std::is_sorted(output_data.begin(), output_data.end()));
}