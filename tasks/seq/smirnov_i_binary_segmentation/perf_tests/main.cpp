#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/smirnov_i_binary_segmentation/include/ops_seq.hpp"

TEST(smirnov_i_binary_segmentation_seq, test_pipeline_run) {
  int cols = 2011;
  int rows = 1193;
  std::vector<int> img;
  img = std::vector<int>(cols * rows, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> mask(rows * cols, 1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  taskDataSeq->outputs_count.emplace_back(cols);
  taskDataSeq->outputs_count.emplace_back(rows);

  auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<int> expected_mask(cols * rows, 2);
  for (int i = 0; i < rows * cols; i++) {
    ASSERT_EQ(expected_mask[i], mask[i]);
  }
}

TEST(smirnov_i_binary_segmentation_seq, test_task_run) {
  int cols = 2011;
  int rows = 1193;
  std::vector<int> img;
  img = std::vector<int>(cols * rows, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> mask(rows * cols, 1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
  taskDataSeq->outputs_count.emplace_back(cols);
  taskDataSeq->outputs_count.emplace_back(rows);

  auto TestTaskSequential = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskSequential>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<int> expected_mask(cols * rows, 2);
  for (int i = 0; i < cols * rows; i++) {
    ASSERT_EQ(expected_mask[i], mask[i]);
  }
}