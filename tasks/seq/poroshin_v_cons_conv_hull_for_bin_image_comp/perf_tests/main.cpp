// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/poroshin_v_cons_conv_hull_for_bin_image_comp/include/ops_seq.hpp"

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_seq, test_pipeline_run) {
  const int m = 1000;
  const int n = 3000;

  std::vector<int> tmp(m * n, 1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  std::vector<std::pair<int, int>> result(m * n + 2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));

  auto TestTaskSequential =
      std::make_shared<poroshin_v_cons_conv_hull_for_bin_image_comp_seq::TestTaskSequential>(taskDataSeq);

  ASSERT_TRUE(TestTaskSequential->validation());
  TestTaskSequential->pre_processing();
  TestTaskSequential->run();
  TestTaskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<std::pair<int, int>> ans = {{0, 0}, {999, 0}, {999, 2999}, {0, 2999}, {0, 0}, {-1, -1}};
  std::vector<std::pair<int, int>> res(ans.size());
  std::copy(result.begin(), result.begin() + ans.size(), res.begin());
  ASSERT_EQ(ans, res);
}

TEST(poroshin_v_cons_conv_hull_for_bin_image_comp_seq, test_task_run) {
  const int m = 1000;
  const int n = 3000;

  std::vector<int> tmp(m * n, 1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  std::vector<std::pair<int, int>> result(m * n + 2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));

  auto TestTaskSequential =
      std::make_shared<poroshin_v_cons_conv_hull_for_bin_image_comp_seq::TestTaskSequential>(taskDataSeq);

  ASSERT_TRUE(TestTaskSequential->validation());
  TestTaskSequential->pre_processing();
  TestTaskSequential->run();
  TestTaskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(TestTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<std::pair<int, int>> ans = {{0, 0}, {999, 0}, {999, 2999}, {0, 2999}, {0, 0}, {-1, -1}};
  std::vector<std::pair<int, int>> res(ans.size());
  std::copy(result.begin(), result.begin() + ans.size(), res.begin());
  ASSERT_EQ(ans, res);
}