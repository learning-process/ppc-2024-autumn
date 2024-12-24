#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kurakin_m_graham_scan/include/kurakin_graham_scan_ops_seq.hpp"

TEST(kurakin_m_graham_scan_seq, test_pipeline_run) {
  int count_point;
  std::vector<double> points;

  // Create data
  count_point = 100000;
  points = std::vector<double>(count_point * 2);

  points[0] = count_point / 2;
  points[1] = (-1) * count_point / 2;

  for (int i = 2; i < count_point * 2; i += 2) {
    points[i] = count_point / 2 - i / 2;
    points[i + 1] = count_point / 2 - i / 2;
  }

  int scan_size;
  std::vector<double> scan_points(count_point * 2, 0);

  int ans_size = count_point;
  std::vector<double> ans(ans_size * 2);

  ans[0] = count_point / 2;
  ans[1] = (-1) * count_point / 2;

  for (int i = 2; i < count_point * 2; i += 2) {
    ans[i] = count_point / 2 - i / 2;
    ans[i + 1] = count_point / 2 - i / 2;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points.data()));
  taskDataSeq->outputs_count.emplace_back(scan_points.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kurakin_m_graham_scan_seq::TestTaskSequential>(taskDataSeq);

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

  EXPECT_EQ(ans_size, scan_size);
  for (int i = 0; i < ans_size * 2; i += 2) {
    EXPECT_EQ(ans[i], scan_points[i]);
    EXPECT_EQ(ans[i + 1], scan_points[i + 1]);
  }
}

TEST(kurakin_m_graham_scan_seq, test_task_run) {
  int count_point;
  std::vector<double> points;

  // Create data
  count_point = 100000;
  points = std::vector<double>(count_point * 2);

  points[0] = count_point / 2;
  points[1] = (-1) * count_point / 2;

  for (int i = 2; i < count_point; i += 2) {
    points[i] = count_point / 2 - i / 2;
    points[i + 1] = count_point / 2 - i / 2;
  }

  int scan_size;
  std::vector<double> scan_points(count_point * 2, 0);

  int ans_size = count_point;
  std::vector<double> ans(ans_size * 2);

  ans[0] = count_point / 2;
  ans[1] = (-1) * count_point / 2;

  for (int i = 2; i < count_point; i += 2) {
    ans[i] = count_point / 2 - i / 2;
    ans[i + 1] = count_point / 2 - i / 2;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  taskDataSeq->inputs_count.emplace_back(points.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_points.data()));
  taskDataSeq->outputs_count.emplace_back(scan_points.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kurakin_m_graham_scan_seq::TestTaskSequential>(taskDataSeq);

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

  EXPECT_EQ(ans_size, scan_size);
  for (int i = 0; i < ans_size * 2; i += 2) {
    EXPECT_EQ(ans[i], scan_points[i]);
    EXPECT_EQ(ans[i + 1], scan_points[i + 1]);
  }
}
