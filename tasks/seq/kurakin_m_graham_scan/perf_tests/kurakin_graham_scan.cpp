#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kurakin_m_graham_scan/include/kurakin_graham_scan_ops_seq.hpp"

TEST(kurakin_m_graham_scan_seq, test_pipeline_run) {
  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;

  // Create data
  count_point = 1000;
  point_x = std::vector<double>(count_point);
  point_y = std::vector<double>(count_point);

  point_x[0] = count_point / 2;
  point_y[0] = (-1) * count_point / 2;

  for (int i = 1; i < count_point; i++) {
    point_x[i] = count_point / 2 - i;
    point_y[i] = count_point / 2 - i;
  }

  int scan_size;
  std::vector<double> scan_x(count_point, 0);
  std::vector<double> scan_y(count_point, 0);

  int ans_size = count_point;
  std::vector<double> ans_x(ans_size);
  std::vector<double> ans_y(ans_size);

  ans_x[0] = ans_size / 2;
  ans_y[0] = (-1) * ans_size / 2;

  for (int i = 1; i < ans_size; i++) {
    ans_x[i] = ans_size / 2 - i;
    ans_y[i] = ans_size / 2 - i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
  taskDataSeq->inputs_count.emplace_back(point_x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
  taskDataSeq->inputs_count.emplace_back(point_y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x.data()));
  taskDataSeq->outputs_count.emplace_back(scan_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y.data()));
  taskDataSeq->outputs_count.emplace_back(scan_y.size());

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
  for (int i = 0; i < ans_size; i++) {
    EXPECT_EQ(ans_x[i], scan_x[i]);
    EXPECT_EQ(ans_y[i], scan_y[i]);
  }
}

TEST(kurakin_m_graham_scan_seq, test_task_run) {
  int count_point;
  std::vector<double> point_x;
  std::vector<double> point_y;

  // Create data
  count_point = 1000;
  point_x = std::vector<double>(count_point);
  point_y = std::vector<double>(count_point);

  point_x[0] = count_point / 2;
  point_y[0] = (-1) * count_point / 2;

  for (int i = 1; i < count_point; i++) {
    point_x[i] = count_point / 2 - i;
    point_y[i] = count_point / 2 - i;
  }

  int scan_size;
  std::vector<double> scan_x(count_point, 0);
  std::vector<double> scan_y(count_point, 0);

  int ans_size = count_point;
  std::vector<double> ans_x(ans_size);
  std::vector<double> ans_y(ans_size);

  ans_x[0] = ans_size / 2;
  ans_y[0] = (-1) * ans_size / 2;

  for (int i = 1; i < ans_size; i++) {
    ans_x[i] = ans_size / 2 - i;
    ans_y[i] = ans_size / 2 - i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_x.data()));
  taskDataSeq->inputs_count.emplace_back(point_x.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(point_y.data()));
  taskDataSeq->inputs_count.emplace_back(point_y.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&scan_size));
  taskDataSeq->outputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_x.data()));
  taskDataSeq->outputs_count.emplace_back(scan_x.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(scan_y.data()));
  taskDataSeq->outputs_count.emplace_back(scan_y.size());

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
  for (int i = 0; i < ans_size; i++) {
    EXPECT_EQ(ans_x[i], scan_x[i]);
    EXPECT_EQ(ans_y[i], scan_y[i]);
  }
}
