#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/guseynov_e_marking_comps_of_bin_image/include/ops_seq.hpp"

TEST(guseynov_e_marking_comps_of_bin_image_seq, test_pipeline_run) {
  const int rows = 1250;
  const int columns = 1250;
  std::vector<int> in(rows * columns);
  std::vector<int> out(rows * columns, -1);
  std::vector<int> expected_out(rows * columns);
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < columns; y++) {
      int pos = x * columns + y;
      if (x < 50) {
        in[pos] = 0;
        expected_out[pos] = 2;
      } else if (x == 50) {
        in[pos] = 1;
        expected_out[pos] = 1;
      } else if (x == 51) {
        in[pos] = 0;
        expected_out[pos] = 3;
      } else if (x == 52) {
        in[pos] = 1;
        expected_out[pos] = 1;
      } else {
        in[pos] = 0;
        expected_out[pos] = 4;
      }
    }
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(rows);
  taskDataSeq->outputs_count.emplace_back(columns);

  // Create Task
  auto testTaskSequential =
      std::make_shared<guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(expected_out, out);
}

TEST(guseynov_e_marking_comps_of_bin_image_seq, test_task_run) {
  const int rows = 1250;
  const int columns = 1250;
  std::vector<int> in(rows * columns);
  std::vector<int> out(rows * columns, -1);
  std::vector<int> expected_out(rows * columns);
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < columns; y++) {
      int pos = x * columns + y;
      if (x < 50) {
        in[pos] = 0;
        expected_out[pos] = 2;
      } else if (x == 50) {
        in[pos] = 1;
        expected_out[pos] = 1;
      } else if (x == 51) {
        in[pos] = 0;
        expected_out[pos] = 3;
      } else if (x == 52) {
        in[pos] = 1;
        expected_out[pos] = 1;
      } else {
        in[pos] = 0;
        expected_out[pos] = 4;
      }
    }
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(rows);
  taskDataSeq->outputs_count.emplace_back(columns);

  // Create Task
  auto testTaskSequential =
      std::make_shared<guseynov_e_marking_comps_of_bin_image_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(expected_out, out);
}