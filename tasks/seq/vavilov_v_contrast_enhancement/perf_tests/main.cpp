#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vavilov_v_contrast_enhancement/include/ops_seq.hpp"

TEST(vavilov_v_contrast_enhancement_seq, test_task_run) {

  size_t data_size = 1000000;
  std::vector<int> input(data_size, 128);
  std::vector<int> output(input.size());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs_count.emplace_back(input.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));

  auto testTaskSequential = std::make_shared<vavilov_v_contrast_enhancement_seq::TestTaskSequential>(taskDataSeq);

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

  for (size_t i = 0; i < data_size; i++) {
    ASSERT_EQ(output[i], 0);
  };
}

TEST(vavilov_v_contrast_enhancement_seq, test_pipeline_run) {
  size_t data_size = 1000000;
  std::vector<int> input(data_size, 128);
  std::vector<int> output(input.size());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs_count.emplace_back(input.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));

  auto testTaskSequential = std::make_shared<vavilov_v_contrast_enhancement_seq::TestTaskSequential>(taskDataSeq);

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

  for (size_t i = 0; i < data_size; i++) {
    ASSERT_EQ(output[i], 0);
  };
}
